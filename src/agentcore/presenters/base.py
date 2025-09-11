import abc
from pathlib import PurePath
from typing import Any

import jinja2


class BasePresenter(abc.ABC):
    @property
    @abc.abstractmethod
    def _template_path(self) -> PurePath:
        return PurePath("presenters")

    def __init__(self, jinja: jinja2.Environment):
        self._jinja: jinja2.Environment = jinja

    async def _render(self, template_name: str, **kwargs: Any):
        final_path = str(self._template_path / template_name)
        template = self._jinja.get_template(final_path)
        return await template.render_async(**kwargs)
