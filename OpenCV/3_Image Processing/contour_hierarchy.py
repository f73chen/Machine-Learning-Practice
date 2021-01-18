# recall when using cv2.findContours() passed retrieval mode
    # ex. cv2.RETR_LIST, cv2.RETR_TREE, etc.
# also, in output got a "hierarchy"
# when shapes are nested, call outer parent and inner child

# external or outermost contours are hierarchy-0
    # aka same hierarchy level
# inside contour of thick frame is child of outside contour of that frame
    # hierarchy-1
# first child: the (arbitrary) highest ID'd child

# array of 4 values: [next, previous, first_child, parent]
# next: next contour at the same hierarchical level
# previous: previous contour at the same hierarchical level
    # if none, that field returns index = -1

# 1. RETR_LIST
# retrieves all contours, but doesn't create parent-child relationships
# parents and children are equal --> last 2 fields = -1
# [[ 1, -1, -1, -1],
#  [ 2,  0, -1, -1],
#  [ 3,  1, -1, -1],
#  [ 4,  2, -1, -1],
#  [ 5,  3, -1, -1],
#  [ 6,  4, -1, -1],
#  [ 7,  5, -1, -1],
#  [-1,  6, -1, -1]]

# 2. RETR_EXTERNAL
# only returns extreme outer flags (hierarchy-0); no child contours
# only the eldest in the family, not other members
# [[ 1, -1, -1, -1],
#  [ 2,  0, -1, -1],
#  [-1,  1, -1, -1]]

# 3. RETR_CCOMP
# retrieves all contours and arranges them to a 2-level hierarchy
# external contours (boundary) in hierarchy-1
# contours of holes inside in hierarchy-2
# if there are any objects inside the hole, contour again in hierarchy-1, then their holes in hierarchy-2
# [[ 3, -1,  1, -1],
#  [ 2, -1, -1,  0],
#  [-1,  1, -1,  0],
#  [ 5,  0,  4, -1],
#  [-1, -1, -1,  3],
#  [ 7,  3,  6, -1],
#  [-1, -1, -1,  5],
#  [ 8,  5, -1, -1],
#  [-1,  7, -1, -1]]

# 4. RETR_TREE
# retrieves all contours and creates full family hierarchy
# [[ 7, -1,  1, -1],
#  [-1, -1,  2,  0],
#  [-1, -1,  3,  1],
#  [-1, -1,  4,  2],
#  [-1, -1,  5,  3],
#  [ 6, -1, -1,  4],
#  [-1,  5, -1,  4],
#  [ 8,  0, -1, -1],
#  [-1,  7, -1, -1]]