import cv2 as cv

img = cv.imread('ratunel.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(
    gray,
    127,
    255,
    cv.THRESH_BINARY
)

_, thresh_inv = cv.threshold(
    gray,
    127,
    255,
    cv.THRESH_BINARY_INV
)

_, truncate_tr = cv.threshold(
    gray,
    100,
    255, #truncate
    cv.THRESH_TRUNC
)
_, to_zero = cv.threshold(
    gray,
    100,
    255,
    cv.THRESH_TOZERO
)
adaptive = cv.adaptiveThreshold(
    gray,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    11,
    2
)

_, otsu = cv.threshold(
    gray,
    0,
    255,
    cv.THRESH_BINARY + cv.THRESH_OTSU
)
cv.imshow('actual ratunel', img)
cv.imshow('binary ratunel', thresh)
cv.imshow('binary inverted ratunel', thresh_inv)
cv.imshow('truncate ratunel', truncate_tr)
cv.imshow('to zero ratunel', to_zero)
cv.imshow('adaptive thresh', adaptive)
cv.imshow('thresh otsu ratunel', otsu)
cv.waitKey(0)
cv.destroyAllWindows()