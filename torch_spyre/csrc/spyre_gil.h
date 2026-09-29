/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <pybind11/pybind11.h>

#include <optional>

namespace spyre {

// Native completion releases retained Tensor/Storage objects before publishing
// completion. Their PyObject-linked refcounts can require the GIL. A caller
// waiting for that completion must let cleanup acquire it. Dispatcher and C++
// callers may already run without the GIL, so release only when it is held.
class ReleaseGilIfHeld {
 public:
  ReleaseGilIfHeld() {
    if (Py_IsInitialized() && PyGILState_Check()) {
      release_.emplace();
    }
  }

 private:
  std::optional<pybind11::gil_scoped_release> release_;
};

}  // namespace spyre
