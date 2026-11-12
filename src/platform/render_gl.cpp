// OpenGL function loading and the point-batch renderer.
// Included by the platform TU after <GL/gl.h> is available. The platform
// provides gl_get_proc(name) (glXGetProcAddressARB / wglGetProcAddress) and a
// current context before calling gl_renderer_init.
//
// We declare the handful of post-1.1 entry points we use ourselves instead of
// depending on glext.h / glcorearb.h, which MSVC does not ship.
// glActiveTexture / glBlendEquation come from gl.h + libGL on Linux; the
// (untested) Windows layer will need them added to the loader.

#ifndef GL_APIENTRY_
#if SIM_OS_WINDOWS
#define GL_APIENTRY_ APIENTRY
#else
#define GL_APIENTRY_
#endif
#endif

typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

#define GL_ARRAY_BUFFER        0x8892
#define GL_STREAM_DRAW         0x88E0
#define GL_FRAGMENT_SHADER     0x8B30
#define GL_VERTEX_SHADER       0x8B31
#define GL_COMPILE_STATUS      0x8B81
#define GL_LINK_STATUS         0x8B82
#define GL_INFO_LOG_LENGTH     0x8B84
#define GL_PROGRAM_POINT_SIZE  0x8642
#define GL_MAJOR_VERSION       0x821B
#define GL_MINOR_VERSION       0x821C
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_TEXTURE0            0x84C0
#define GL_FUNC_ADD            0x8006
#define GL_CLAMP_TO_EDGE       0x812F

#define GL_FUNCTION_LIST(X) \
    X(GLuint, glCreateShader, GLenum type) \
    X(void, glShaderSource, GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length) \
    X(void, glCompileShader, GLuint shader) \
    X(void, glGetShaderiv, GLuint shader, GLenum pname, GLint *params) \
    X(void, glGetShaderInfoLog, GLuint shader, GLsizei max_len, GLsizei *length, GLchar *log) \
    X(void, glDeleteShader, GLuint shader) \
    X(GLuint, glCreateProgram, void) \
    X(void, glAttachShader, GLuint program, GLuint shader) \
    X(void, glLinkProgram, GLuint program) \
    X(void, glGetProgramiv, GLuint program, GLenum pname, GLint *params) \
    X(void, glGetProgramInfoLog, GLuint program, GLsizei max_len, GLsizei *length, GLchar *log) \
    X(void, glUseProgram, GLuint program) \
    X(GLint, glGetUniformLocation, GLuint program, const GLchar *name) \
    X(void, glUniformMatrix4fv, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
    X(void, glUniform1f, GLint location, GLfloat v0) \
    X(void, glUniform4fv, GLint location, GLsizei count, const GLfloat *value) \
    X(void, glGenBuffers, GLsizei n, GLuint *buffers) \
    X(void, glBindBuffer, GLenum target, GLuint buffer) \
    X(void, glBufferData, GLenum target, GLsizeiptr size, const void *data, GLenum usage) \
    X(void, glGenVertexArrays, GLsizei n, GLuint *arrays) \
    X(void, glBindVertexArray, GLuint array) \
    X(void, glEnableVertexAttribArray, GLuint index) \
    X(void, glVertexAttribPointer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) \
    X(void, glUniform1i, GLint location, GLint v0) \
    X(void, glBlendFuncSeparate, GLenum src_rgb, GLenum dst_rgb, GLenum src_a, GLenum dst_a) \
    X(void, glDrawElementsBaseVertex, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint base_vertex)

#define GL_DECLARE_FUNCTION(ret, name, ...) typedef ret (GL_APIENTRY_ *name##_fn)(__VA_ARGS__); global name##_fn name;
GL_FUNCTION_LIST(GL_DECLARE_FUNCTION)
#undef GL_DECLARE_FUNCTION

typedef void *gl_get_proc_fn(const char *name);

internal b32 gl_load_functions(gl_get_proc_fn *get_proc) {
    b32 ok = 1;
#define GL_LOAD_FUNCTION(ret, name, ...) \
    name = (name##_fn)get_proc(#name); \
    if (!name) { fprintf(stderr, "[gl] missing %s\n", #name); ok = 0; }
    GL_FUNCTION_LIST(GL_LOAD_FUNCTION)
#undef GL_LOAD_FUNCTION
    return ok;
}

// ----------------------------------------------------------------------------

#define GL_MAX_UI_TEXTURES 64

struct GLUiTexture {
    u32 id;               // RenderTexOp id (0 = free slot)
    GLuint gl;
};

struct GLRenderer {
    GLuint program;
    GLuint vao;
    GLuint vbo[3];        // one buffer per SoA axis: x, y, z
    GLint u_view_proj;
    GLint u_point_size;
    GLint u_color;

    GLuint ui_program;
    GLuint ui_vao, ui_vbo, ui_ibo;
    GLint u_ui_proj, u_ui_tex;
    GLUiTexture ui_textures[GL_MAX_UI_TEXTURES];
};

global const char *gl_ui_vertex_src =
    "#version 330 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec2 uv;\n"
    "layout(location = 2) in vec4 col;\n"
    "uniform mat4 u_proj;\n"
    "out vec2 f_uv;\n"
    "out vec4 f_col;\n"
    "void main() {\n"
    "    f_uv = uv;\n"
    "    f_col = col;\n"
    "    gl_Position = u_proj * vec4(pos, 0.0, 1.0);\n"
    "}\n";

global const char *gl_ui_fragment_src =
    "#version 330 core\n"
    "uniform sampler2D u_tex;\n"
    "in vec2 f_uv;\n"
    "in vec4 f_col;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    frag_color = f_col * texture(u_tex, f_uv);\n"
    "}\n";

global const char *gl_point_vertex_src =
    "#version 330 core\n"
    "layout(location = 0) in float px;\n"
    "layout(location = 1) in float py;\n"
    "layout(location = 2) in float pz;\n"
    "uniform mat4 u_view_proj;\n"
    "uniform float u_point_size;\n"
    "void main() {\n"
    "    gl_Position = u_view_proj * vec4(px, py, pz, 1.0);\n"
    "    gl_PointSize = u_point_size;\n"
    "}\n";

global const char *gl_point_fragment_src =
    "#version 330 core\n"
    "uniform vec4 u_color;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    vec2 d = gl_PointCoord * 2.0 - 1.0;\n"
    "    float r2 = dot(d, d);\n"
    "    if (r2 > 1.0) discard;\n"
    "    frag_color = vec4(u_color.rgb * (1.0 - 0.35 * r2), u_color.a);\n"   // soft round dots
    "}\n";

internal GLuint gl_compile_shader(GLenum type, const char *src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, 0);
    glCompileShader(shader);
    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), 0, log);
        fprintf(stderr, "[gl] shader compile failed:\n%s\n", log);
        return 0;
    }
    return shader;
}

internal GLuint gl_link_program(const char *vs_src, const char *fs_src) {
    GLuint vs = gl_compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = gl_compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), 0, log);
        fprintf(stderr, "[gl] program link failed:\n%s\n", log);
        return 0;
    }
    return program;
}

internal b32 gl_ui_init(GLRenderer *r) {
    r->ui_program = gl_link_program(gl_ui_vertex_src, gl_ui_fragment_src);
    if (!r->ui_program) return 0;
    r->u_ui_proj = glGetUniformLocation(r->ui_program, "u_proj");
    r->u_ui_tex = glGetUniformLocation(r->ui_program, "u_tex");
    glGenVertexArrays(1, &r->ui_vao);
    glGenBuffers(1, &r->ui_vbo);
    glGenBuffers(1, &r->ui_ibo);
    glBindVertexArray(r->ui_vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->ui_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->ui_ibo);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(RenderUiVertex), (void *)OffsetOf(RenderUiVertex, x));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(RenderUiVertex), (void *)OffsetOf(RenderUiVertex, u));
    glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(RenderUiVertex), (void *)OffsetOf(RenderUiVertex, rgba));
    glBindVertexArray(0);
    return 1;
}

// After a sim hot reload the UI is rebuilt with fresh texture ids.
internal void gl_drop_ui_textures(GLRenderer *r) {
    for (u32 i = 0; i < GL_MAX_UI_TEXTURES; ++i) {
        if (r->ui_textures[i].id) glDeleteTextures(1, &r->ui_textures[i].gl);
        r->ui_textures[i].id = 0;
    }
}

internal GLUiTexture *gl_find_ui_texture(GLRenderer *r, u32 id) {
    for (u32 i = 0; i < GL_MAX_UI_TEXTURES; ++i) if (r->ui_textures[i].id == id) return r->ui_textures + i;
    return 0;
}

internal void gl_apply_tex_ops(GLRenderer *r, RenderCommands *cmds) {
    for (u32 i = 0; i < cmds->tex_op_count && i < RENDER_MAX_TEX_OPS; ++i) {
        RenderTexOp *op = cmds->tex_ops + i;
        GLUiTexture *t = gl_find_ui_texture(r, op->id);
        if (op->kind == RenderTexOp_Destroy) {
            if (t) { glDeleteTextures(1, &t->gl); t->id = 0; }
            continue;
        }
        if (!t) {
            t = gl_find_ui_texture(r, 0);
            if (!t) { fprintf(stderr, "[gl] out of UI texture slots\n"); continue; }
            t->id = op->id;
            glGenTextures(1, &t->gl);
        }
        glBindTexture(GL_TEXTURE_2D, t->gl);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)op->width, (GLsizei)op->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, op->rgba);
    }
}

internal void gl_render_ui(GLRenderer *r, RenderCommands *cmds, u32 width, u32 height) {
    if (!cmds->ui_cmd_count || !r->ui_program) return;
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
    glUseProgram(r->ui_program);
    f32 L = 0.0f, R = (f32)width, T = 0.0f, B = (f32)height;
    f32 proj[16] = {
        2.0f / (R - L), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / (T - B), 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        (R + L) / (L - R), (T + B) / (B - T), 0.0f, 1.0f,
    };
    glUniformMatrix4fv(r->u_ui_proj, 1, GL_FALSE, proj);
    glUniform1i(r->u_ui_tex, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(r->ui_vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->ui_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)cmds->ui_vtx_count * (GLsizeiptr)sizeof(RenderUiVertex), cmds->ui_vtx, GL_STREAM_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->ui_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)cmds->ui_idx_count * (GLsizeiptr)sizeof(u32), cmds->ui_idx, GL_STREAM_DRAW);
    for (u32 c = 0; c < cmds->ui_cmd_count; ++c) {
        const RenderUiCmd *cmd = cmds->ui_cmds + c;
        f32 x0 = Maximum(cmd->clip_x0, 0.0f), y0 = Maximum(cmd->clip_y0, 0.0f);
        f32 x1 = Minimum(cmd->clip_x1, (f32)width), y1 = Minimum(cmd->clip_y1, (f32)height);
        if (x1 <= x0 || y1 <= y0) continue;
        glScissor((GLint)x0, (GLint)((f32)height - y1), (GLsizei)(x1 - x0), (GLsizei)(y1 - y0));
        GLUiTexture *t = gl_find_ui_texture(r, cmd->texture);
        glBindTexture(GL_TEXTURE_2D, t ? t->gl : 0);
        glDrawElementsBaseVertex(GL_TRIANGLES, (GLsizei)cmd->elem_count, GL_UNSIGNED_INT,
                                 (void *)((u64)cmd->idx_offset * sizeof(u32)), (GLint)cmd->vtx_offset);
    }
    glBindVertexArray(0);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

internal b32 gl_renderer_init(GLRenderer *r) {
    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    printf("[gl] %s | %s | GL %d.%d\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER), major, minor);

    GLuint vs = gl_compile_shader(GL_VERTEX_SHADER, gl_point_vertex_src);
    GLuint fs = gl_compile_shader(GL_FRAGMENT_SHADER, gl_point_fragment_src);
    if (!vs || !fs) return 0;
    r->program = glCreateProgram();
    glAttachShader(r->program, vs);
    glAttachShader(r->program, fs);
    glLinkProgram(r->program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint status = 0;
    glGetProgramiv(r->program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[2048];
        glGetProgramInfoLog(r->program, sizeof(log), 0, log);
        fprintf(stderr, "[gl] program link failed:\n%s\n", log);
        return 0;
    }
    if (!gl_ui_init(r)) return 0;
    r->u_view_proj = glGetUniformLocation(r->program, "u_view_proj");
    r->u_point_size = glGetUniformLocation(r->program, "u_point_size");
    r->u_color = glGetUniformLocation(r->program, "u_color");

    glGenVertexArrays(1, &r->vao);
    glBindVertexArray(r->vao);
    glGenBuffers(3, r->vbo);
    for (u32 axis = 0; axis < 3; ++axis) {
        glBindBuffer(GL_ARRAY_BUFFER, r->vbo[axis]);
        glEnableVertexAttribArray(axis);
        glVertexAttribPointer(axis, 1, GL_FLOAT, GL_FALSE, 0, 0);
    }
    glBindVertexArray(0);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    return 1;
}

internal void gl_render_frame(GLRenderer *r, RenderCommands *cmds, u32 width, u32 height) {
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
    glDisable(GL_SCISSOR_TEST);
    glClearColor(cmds->clear_color.r, cmds->clear_color.g, cmds->clear_color.b, cmds->clear_color.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    gl_apply_tex_ops(r, cmds);

    if (cmds->batch_count) {
        if (cmds->scene_w > 0.0f && cmds->scene_h > 0.0f) {
            GLint x = (GLint)cmds->scene_x, y = (GLint)((f32)height - cmds->scene_y - cmds->scene_h);
            glViewport(x, y, (GLsizei)cmds->scene_w, (GLsizei)cmds->scene_h);
            glScissor(x, y, (GLsizei)cmds->scene_w, (GLsizei)cmds->scene_h);
            glEnable(GL_SCISSOR_TEST);
        }
        glEnable(GL_DEPTH_TEST);
        glUseProgram(r->program);
        glUniformMatrix4fv(r->u_view_proj, 1, GL_FALSE, &cmds->view_proj.e[0][0]);
        glBindVertexArray(r->vao);
        for (u32 b = 0; b < cmds->batch_count && b < RENDER_MAX_BATCHES; ++b) {
            RenderPointBatch *batch = cmds->batches + b;
            if (!batch->count || !batch->px) continue;
            glUniform1f(r->u_point_size, batch->point_size);
            glUniform4fv(r->u_color, 1, batch->color.e);
            GLsizeiptr bytes = (GLsizeiptr)batch->count * (GLsizeiptr)sizeof(f32);
            const f32 *axes[3] = { batch->px, batch->py, batch->pz };
            for (u32 axis = 0; axis < 3; ++axis) {
                glBindBuffer(GL_ARRAY_BUFFER, r->vbo[axis]);
                glBufferData(GL_ARRAY_BUFFER, bytes, axes[axis], GL_STREAM_DRAW);   // orphan + upload
            }
            glDrawArrays(GL_POINTS, 0, (GLsizei)batch->count);
        }
        glBindVertexArray(0);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, (GLsizei)width, (GLsizei)height);
    }
    gl_render_ui(r, cmds, width, height);
}

// Reads the back buffer and writes a binary PPM (P6) through the platform
// file API. Debug aid for headless CI / remote machines.
internal b32 gl_write_screenshot_ppm(PlatformAPI *api, Arena *scratch, const char *path, u32 width, u32 height) {
    TempMemory temp = begin_temp(scratch);
    char header[64];
    int header_len = snprintf(header, sizeof(header), "P6\n%u %u\n255\n", width, height);
    u64 pixel_bytes = (u64)width * (u64)height * 3;
    u8 *file = (u8 *)push_size_(scratch, (u64)header_len + pixel_bytes, 16, 0);
    u8 *pixels = (u8 *)push_size_(scratch, pixel_bytes, 16, 0);
    memcpy(file, header, (u64)header_len);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, (GLsizei)width, (GLsizei)height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    for (u32 y = 0; y < height; ++y) {   // GL rows are bottom-up
        memcpy(file + header_len + (u64)y * width * 3, pixels + (u64)(height - 1 - y) * width * 3, (u64)width * 3);
    }
    b32 ok = api->write_entire_file(path, file, (u64)header_len + pixel_bytes);
    end_temp(temp);
    return ok;
}
