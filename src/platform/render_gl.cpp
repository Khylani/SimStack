// OpenGL function loading and the milestone-1 point renderer.
// Included by the platform TU after <GL/gl.h> is available. The platform
// provides gl_get_proc(name) (glXGetProcAddressARB / wglGetProcAddress) and a
// current context before calling gl_renderer_init.
//
// We declare the handful of post-1.1 entry points we use ourselves instead of
// depending on glext.h / glcorearb.h, which MSVC does not ship.

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
    X(void, glVertexAttribPointer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)

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

struct GLRenderer {
    GLuint program;
    GLuint vao;
    GLuint vbo[3];        // one buffer per SoA axis: x, y, z
    GLint u_view_proj;
    GLint u_point_size;
    GLint u_color;
};

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
    "    frag_color = u_color;\n"
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
    glClearColor(cmds->clear_color.r, cmds->clear_color.g, cmds->clear_color.b, cmds->clear_color.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!cmds->point_count || !cmds->px) return;

    glUseProgram(r->program);
    glUniformMatrix4fv(r->u_view_proj, 1, GL_FALSE, &cmds->view_proj.e[0][0]);
    glUniform1f(r->u_point_size, cmds->point_size);
    glUniform4fv(r->u_color, 1, cmds->point_color.e);

    glBindVertexArray(r->vao);
    GLsizeiptr bytes = (GLsizeiptr)cmds->point_count * (GLsizeiptr)sizeof(f32);
    const f32 *axes[3] = { cmds->px, cmds->py, cmds->pz };
    for (u32 axis = 0; axis < 3; ++axis) {
        glBindBuffer(GL_ARRAY_BUFFER, r->vbo[axis]);
        glBufferData(GL_ARRAY_BUFFER, bytes, axes[axis], GL_STREAM_DRAW);   // orphan + upload
    }
    glDrawArrays(GL_POINTS, 0, (GLsizei)cmds->point_count);
    glBindVertexArray(0);
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
