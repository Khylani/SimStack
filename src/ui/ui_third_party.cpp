// Dear ImGui + ImPlot as one translation unit. Compiled once into
// build/ui_third_party.o (build.sh rebuilds it only when third_party/ or the
// config changes) and linked into the sim module, so a sim hot reload does
// not recompile ~70k lines. Their globals therefore live in the sim module and
// are re-initialised on every reload (see sim_before_unload / ui_begin).

#include "imgui.cpp"
#include "imgui_draw.cpp"
#include "imgui_widgets.cpp"
#include "imgui_tables.cpp"
#include "implot.cpp"
#include "implot_items.cpp"
