


def initFun():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glShadeModel(GL_FLAT)
 
def displayFun():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 1.0, 1.0)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    glScalef(1.0, 2.0, 1.0)
    glutWireCube(1.0)
    glFlush()
 
def reshapeFun(w,h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
    glMatrixMode(GL_MODELVIEW)
 
if __name__=='__main__':
    glutInit()
    glutInitWindowSize(500, 500)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Cube")
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutDisplayFunc(displayFun)
    glutReshapeFunc(reshapeFun)
    initFun()
    glutMainLoop()