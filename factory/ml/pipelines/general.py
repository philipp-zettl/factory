class PipelineMixin:
    output_type = None

    def get_task(self, is_multi):
        raise NotImplementedError

    def run_task(self, task):
        raise NotImplementedError

    def get_options(self) -> dict:
        raise NotImplementedError
