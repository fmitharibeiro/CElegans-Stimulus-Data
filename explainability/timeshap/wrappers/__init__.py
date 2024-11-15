#  Copyright 2022 Feedzai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .base_wrapper import TimeSHAPWrapper


# Guarding against torch not installed
from ..utils.compatibility import is_torch_installed
if is_torch_installed():
    from .torch_wrappers import TorchModelWrapper

# Guarding against torch not installed
from ..utils.compatibility import is_tensorflow_installed
if is_tensorflow_installed():
    from .tf_wrappers import TensorFlowModelWrapper
