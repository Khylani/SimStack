# Third-party code

| Library | Version | Licence | Where | How it is used |
| --- | --- | --- | --- | --- |
| [Dear ImGui](https://github.com/ocornut/imgui) | v1.93.0-docking (`b916186e66aa3c02b72c6ca17455516512ed8049`) | MIT (`third_party/imgui/LICENSE.txt`) | `third_party/imgui/` (core files only) | Immediate-mode UI, built into the sim module by `src/ui/ui_third_party.cpp` |
| [ImPlot](https://github.com/epezent/implot) | v1.1 (`819f200639455a58b6b3d4156b70e4271aeb2d00`) | MIT (`third_party/implot/LICENSE`) | `third_party/implot/` | Plots in the UI |

Both are compiled with `IMGUI_USER_CONFIG="ui/imconfig_simstack.h"`: no default
allocators (memory comes from the platform), no file IO, 32-bit draw indices.
Update by replacing the files from the pinned tags and recording the new commit here.
