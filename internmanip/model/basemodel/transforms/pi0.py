# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np




def collator_pi0(features):
    batch = {}
    
    for k in features[0].keys():
        if type(features[0][k]) == list:
            batch[k] = [f[k] if k in f else '' for f in features]
        elif type(features[0][k]) == np.ndarray:
            batch[k] = np.stack([f[k] for f in features])
        else:
            batch[k] = torch.stack([f[k] for f in features])
            
    return batch
