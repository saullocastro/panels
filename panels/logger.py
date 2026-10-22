"""Logging functions used by panels

The messages are printed to the standard output, indented with one tab per
level, and returned. Every function has a ``silent`` flag that suppresses the
printing.

"""
def warn(msg, level=0, silent=False):
    """Print a warning message, starting with ``WARNING:``

    The prefix ``WARNING: `` is added to ``msg``, and the result is printed
    after ``level`` tabs unless ``silent`` is ``True``.

    Parameters
    ----------
    msg : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab. Default ``0``.
    silent : bool, optional
        If ``True`` the message is not printed. Default ``False``.

    Returns
    -------
    msg : str
        The message with its prefix, without the indentation.

    """
    msg = 'WARNING: ' + msg
    if not silent:
        print('\t'*level + msg)
    return msg

def error(msg, level=0, silent=False):
    """Print an error message, starting with ``ERROR:``

    The prefix ``ERROR: `` is added to ``msg``, and the result is printed
    after ``level`` tabs unless ``silent`` is ``True``. No exception is
    raised.

    Parameters
    ----------
    msg : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab. Default ``0``.
    silent : bool, optional
        If ``True`` the message is not printed. Default ``False``.

    Returns
    -------
    msg : str
        The message with its prefix, without the indentation.

    """
    msg = 'ERROR: ' + msg
    if not silent:
        print('\t'*level + msg)
    return msg

def msg(msg, level=0, silent=False):
    """Print a log message

    ``msg`` is printed after ``level`` tabs unless ``silent`` is ``True``.

    Parameters
    ----------
    msg : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab. Default ``0``.
    silent : bool, optional
        If ``True`` the message is not printed. Default ``False``.

    Returns
    -------
    msg : str
        The message, without the indentation.

    """
    if not silent:
        print('\t'*level + msg)
    return msg

# def img_popup(filename):
    
#     # To open pop up images - Ignore the syntax warning :)
#     # %matplotlib qt 
#     # For inline images
#     # %matplotlib inline
    
#     plt.figure()
#     plt.title(filename)
#     image = img.imread(filename)
#     plt.imshow(image)
#     plt.show()

