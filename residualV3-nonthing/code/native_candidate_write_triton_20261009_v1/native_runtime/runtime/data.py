

def validate_example(row):
    ids, mask = row['input_ids'], row['loss_mask']
    if len(ids) != len(mask) or not 8 <= len(ids) <= 8192:
        return False
    if any(type(i) is not int or i < 0 or i >= 151936 for i in ids):
        return False
    if any(x not in (0, 1) for x in mask) or 1 not in mask:
        return False
    start = mask.index(1)
    # Whole existing response, never sample a new answer or silently truncate it.
    return start > 0 and all(mask[start:]) and len(ids)-start >= 8
