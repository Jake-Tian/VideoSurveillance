import json

def parse_list_string(s: str):
    """
    Parse a Python-like list string where items may be wrapped in single or double quotes,
    may contain unescaped inner quotes (like man's), commas, and backslashes.
    Returns a list of Python strings.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")

    s = s.strip()
    # Remove outer square brackets if present
    if s.startswith("[") and s.endswith("]"):
        s_body = s[1:-1]
    else:
        s_body = s

    items = []
    cur = []
    i = 0
    n = len(s_body)

    in_single = False
    in_double = False
    escape = False

    def next_non_space(idx):
        # return next non-space char index and char, or (None, None) if none
        j = idx
        while j < n and s_body[j].isspace():
            j += 1
        return (j, s_body[j] if j < n else None)

    while i < n:
        ch = s_body[i]

        if escape:
            # keep the backslash and the escaped char as-is
            cur.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            # start escape (store backslash so we preserve it)
            cur.append(ch)
            escape = True
            i += 1
            continue

        # Handle entering/exiting single-quoted string
        if ch == "'" and not in_double:
            if not in_single:
                # entering single-quoted string: do not include the outer quote
                in_single = True
                i += 1
                continue
            else:
                # potential closing single quote -- decide if it's actually closing
                j, nc = next_non_space(i+1)
                if nc in (",", "]", None):
                    # closing quote
                    in_single = False
                    i += 1
                    continue
                else:
                    # treat as literal apostrophe inside content
                    cur.append(ch)
                    i += 1
                    continue

        # Handle entering/exiting double-quoted string
        if ch == '"' and not in_single:
            if not in_double:
                in_double = True
                i += 1
                continue
            else:
                j, nc = next_non_space(i+1)
                if nc in (",", "]", None):
                    in_double = False
                    i += 1
                    continue
                else:
                    cur.append(ch)
                    i += 1
                    continue

        # Split on commas only if we're not inside any quoted string
        if ch == "," and not in_single and not in_double:
            item = "".join(cur).strip()
            # Only append non-empty items (ignore stray empty pieces)
            items.append(item)
            cur = []
            i += 1
            continue

        # Regular character
        cur.append(ch)
        i += 1

    # add last item
    last = "".join(cur).strip()
    if last != "":
        items.append(last)

    # Now items contains raw textual items, but may still include stray unbalanced quotes.
    # We'll normalize each item into a proper Python string by:
    # - If item starts and ends with a matching quote, strip them (they might be left if parser failed)
    # - Otherwise keep as-is.
    cleaned = []
    for it in items:
        if len(it) >= 2 and ((it[0] == it[-1] == "'") or (it[0] == it[-1] == '"')):
            cleaned.append(it[1:-1])
        else:
            cleaned.append(it)

    return cleaned


def remove_python_code(s: str):
    return s.replace("```", "").replace("python", "").replace("\n", "")



if __name__ == "__main__":
    s = """["WARNING: A man in a blue shirt demands money from a cashier at a convenience store, pointing a gun at the cashier.", 'The cashier, a woman with long hair, appears scared and tries to comply with the man's demands.', "Another man in a black hoodie enters the scene, also pointing a gun at the cashier.", 'The cashier quickly hands over money from a bag to the man in the blue shirt.', 'The man in the black hoodie then forces the cashier to open a cash register.', 'The man in the blue shirt takes the money from the cashier and leaves the store.', 'The cashier remains in the store, looking shaken.', 'The man in the black hoodie also leaves the store.']"""
    print(parse_list_string(s))