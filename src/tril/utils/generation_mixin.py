from typing import Union

from transformers import GenerationMixin
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)


class GenerationMixinWithLogitProcessorSwitch(GenerationMixin):
    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = (
                        "stopping criteria"
                        if isinstance(custom, StoppingCriteria)
                        else "logits processor"
                    )
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to `generate`, "  # noqa
                        f"but it has already been created with the values {default}. {default} has been created by passing the "  # noqa
                        "corresponding arguments to generate or by the model's config default values. "  # noqa
                        f"If you just want to change the default values of {object_type} consider passing them as arguments "  # noqa
                        f"to `generate` instead of using a custom {object_type}."
                    )
        custom_list.extend(default_list)
        return custom_list


def override_generation_routines(cls):
    if cls == GenerationMixinWithLogitProcessorSwitch:
        return cls
    bases = list(cls.__bases__)
    for base_ix in range(len(bases)):
        if (
            bases[base_ix] == GenerationMixin
            and bases[base_ix] != GenerationMixinWithLogitProcessorSwitch
        ):
            bases[base_ix] = GenerationMixinWithLogitProcessorSwitch

        # recursively look up
        if bases[base_ix] != object:
            bases[base_ix] = override_generation_routines(bases[base_ix])

    cls.__bases__ = tuple(bases)
    return cls
