ITEM_TYPES = {
    "SCROLL": 1,
    "PRODUCT": 2,
}

EVENT_TYPES = {
    "QUALIFIED_VIEW": 1,
    "COMPLETE": 2,
    "SKIP": 3,
    "REWATCH": 4,
    "PRODUCT_VIEW": 5,
    "PRODUCT_CLICK": 6,
    "ADD_TO_CART": 7,
    "PURCHASE": 8,
    "SAVE": 9,
    "GRID_IMPRESSION": 10,
    "GRID_DWELL": 11,
    "PRODUCT_HOVER": 12,
    "SCROLL_PAUSE": 13,
    "CHECKOUT_START": 14,
    "SHARE": 15,
}

EVENT_NAMES = {value: name for name, value in EVENT_TYPES.items()}
