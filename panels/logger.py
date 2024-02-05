def warn(msg, level=0, silent=False):
    msg = 'WARNING: ' + msg
    if not silent:
        print('\t'*level + msg)
    return msg

def error(msg, level=0, silent=False):
    msg = 'ERROR: ' + msg
    if not silent:
        print('\t'*level + msg)
    return msg

def msg(msg, level=0, silent=False):
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

