// Dear ImGui compile-time configuration for Simstack (IMGUI_USER_CONFIG).
// Every TU that includes imgui.h gets this through build.sh, so the layout of
// ImGui's types (ImDrawIdx!) is identical in the sim module and the platform.
#pragma once

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DISABLE_DEFAULT_ALLOCATORS          // memory comes from PlatformAPI::allocate (set every frame)
#define IMGUI_DISABLE_FILE_FUNCTIONS              // layout is saved to memory, not imgui.ini
#define IMGUI_DISABLE_DEFAULT_SHELL_FUNCTIONS
#define IMGUI_DISABLE_WIN32_FUNCTIONS

// 32-bit indices: plots with many points never need a vertex-offset split.
#define ImDrawIdx unsigned int
