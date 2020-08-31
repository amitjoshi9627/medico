import engine


def get_prediction(img):
    result = engine.get_results(img)
    if result == 0:
        return "Healthy Plant"
    elif result == 1:
        return "Multiple Diseases Found"
    elif result == 2:
        return "Rust"
    else:
        return "Scab"
