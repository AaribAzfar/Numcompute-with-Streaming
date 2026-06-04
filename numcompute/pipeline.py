import numpy as np


class Pipeline:

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        self.steps = steps

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _intermediate_steps(self):
        return self.steps[:-1]

    def _final_step(self):
        return self.steps[-1]

    def _transform_through(self, X):
        for name, step in self._intermediate_steps():
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Intermediate step '{name}' must implement transform()."
                )
            X = step.transform(X)
        return X

    # ------------------------------------------------------------------ #
    # Batch API
    # ------------------------------------------------------------------ #

    def fit(self, X, y=None):
        X = np.asarray(X)

        # Fit + transform all intermediate steps
        for name, step in self._intermediate_steps():
            if not hasattr(step, "fit"):
                raise AttributeError(f"Step '{name}' must implement fit().")
            step.fit(X)
            X = step.transform(X)

        # Fit the final step
        final_name, final_step = self._final_step()
        if not hasattr(final_step, "fit"):
            raise AttributeError(f"Final step '{final_name}' must implement fit().")

        if y is None:
            final_step.fit(X)
        else:
            final_step.fit(X, np.asarray(y))

        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        final_name, final_step = self._final_step()

        if not hasattr(final_step, "transform"):
            raise AttributeError(
                f"Final step '{final_name}' does not implement transform(); "
                "use fit() + predict() for estimator pipelines."
            )
        X_transformed = self._transform_through(np.asarray(X))
        return final_step.transform(X_transformed)

    # ------------------------------------------------------------------ #
    # Streaming API
    # ------------------------------------------------------------------ #

    def partial_fit(self, X, y=None):
        X = np.asarray(X)
        X_current = X.copy()

        # Update + transform all intermediate steps
        for name, step in self._intermediate_steps():
            if hasattr(step, "partial_fit"):
                step.partial_fit(X_current)
            elif hasattr(step, "fit"):
                step.fit(X_current)
            else:
                raise AttributeError(
                    f"Intermediate step '{name}' must implement partial_fit() or fit()."
                )

            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Intermediate step '{name}' must implement transform()."
                )
            X_current = step.transform(X_current)

        # Update the final step
        final_name, final_step = self._final_step()

        if hasattr(final_step, "partial_fit"):
            if y is None:
                final_step.partial_fit(X_current)
            else:
                final_step.partial_fit(X_current, np.asarray(y))

        elif hasattr(final_step, "fit"):
            if y is None:
                final_step.fit(X_current)
            else:
                final_step.fit(X_current, np.asarray(y))

        else:
            raise AttributeError(
                f"Final step '{final_name}' must implement partial_fit() or fit()."
            )

        return self

    # ------------------------------------------------------------------ #
    # Inference API
    # ------------------------------------------------------------------ #

    def transform(self, X):
        X = np.asarray(X)
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Step '{name}' does not implement transform(). "
                    "Use predict() for pipelines ending in a model."
                )
            X = step.transform(X)
        return X

    def predict(self, X):
        X = np.asarray(X)
        X_transformed = self._transform_through(X)

        final_name, final_step = self._final_step()

        # Final step is a model → use predict
        if hasattr(final_step, "predict"):
            return final_step.predict(X_transformed)

        # Final step is a transformer → fall back to transform
        if hasattr(final_step, "transform"):
            return final_step.transform(X_transformed)

        raise AttributeError(
            f"Final step '{final_name}' must implement predict() or transform()."
        )

    def predict_proba(self, X):
        X = np.asarray(X)
        X_transformed = self._transform_through(X)

        final_name, final_step = self._final_step()

        if not hasattr(final_step, "predict_proba"):
            raise AttributeError(
                f"Final step '{final_name}' does not implement predict_proba()."
            )
        return final_step.predict_proba(X_transformed)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_step(self, name):
        for n, step in self.steps:
            if n == name:
                return step
        raise KeyError(f"No step named '{name}' in pipeline.")

    def __repr__(self):
        step_str = "\n  ".join(f"({name}): {type(step).__name__}" for name, step in self.steps)
        return f"Pipeline(\n  {step_str}\n)"