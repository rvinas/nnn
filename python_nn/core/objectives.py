"""
objectives.py: Objective (or loss) functions. They require its derivatives
               with respect to the network prediction.
Copyright 2017 Ramon Vinas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math


class Objective:
    def __init__(self):
        raise NotImplementedError

    def get_loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class CrossEntropy(Objective):
    def __init__(self):
        pass

    def get_loss(self, y_true, y_pred):
        return -y_true * math.log(y_pred) - (1 - y_true) * math.log(1 - y_pred)

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
