class PipelineMixin:
    output_type = None
    requires_model_load = True

    def _load_pipeline(self):
        raise NotImplementedError

    def get_task(self, is_multi):
        raise NotImplementedError

    def run_task(self, task):
        raise NotImplementedError

    def get_options(self) -> dict:
        raise NotImplementedError
