"""Remove only unused cameras, with an AST proof; original repo stays unchanged."""
import ast
import inspect
import textwrap
from common import RUN


def install_top_only(directory=None):
    import sim_env
    original = textwrap.dedent(inspect.getsource(sim_env.BimanualViperXTask.get_observation))
    lines = ["    obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')\n",
             "    obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')\n"]
    source = original
    for line in lines:
        assert source.count(line) == 1
        source = source.replace(line, '')
    before, after = ast.parse(original), ast.parse(source)
    allowed = [ast.dump(ast.parse(line.strip()).body[0]) for line in lines]
    before.body[0].body = [node for node in before.body[0].body if ast.dump(node) not in allowed]
    assert ast.dump(before) == ast.dump(after)
    namespace = dict(sim_env.__dict__)
    exec(compile(source, '<v2_verified_top_camera>', 'exec'), namespace)
    sim_env.BimanualViperXTask.get_observation = namespace['get_observation']
    if directory:
        (directory/'observation_original.py').write_text(original)
        (directory/'observation_top_only.py').write_text(source)


def identity(physics):
    from OpenGL import GL
    def query():
        out = {}
        for name in ['GL_VENDOR', 'GL_RENDERER', 'GL_VERSION']:
            value = GL.glGetString(getattr(GL, name))
            assert value, 'No current OpenGL context'
            out[name] = value.decode()
        out['gl_error'] = int(GL.glGetError())
        assert out['gl_error'] == 0
        return out
    with physics.contexts.gl.make_current() as context:
        result = context.call(query)
    result['context_class'] = type(physics.contexts.gl).__module__+'.'+type(physics.contexts.gl).__name__
    return result
