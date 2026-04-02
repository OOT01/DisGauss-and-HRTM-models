# =========================================================
# HELPER PRINT FUNCTIONS (FOR TERMINAL OUTPUT)
# =========================================================

def print_section(title, color="", reset="\033[0m"):
    line = "=" * 42

    if color:
        print(f"{color}\n{line}")
        print(f"{title:^42}")
        print(f"{line}{reset}")
    else:
        print(f"\n{line}")
        print(f"{title:^42}")
        print(line)


def print_subsection(title):
    print(f"\n--- {title} ---")


def print_param(name, value, unit_text="", float_format=".6e"):
    if value is None:
        value_str = "N/A"
    elif isinstance(value, float):
        try:
            value_str = format(value, float_format)
        except Exception:
            value_str = str(value)
    else:
        value_str = str(value)

    if unit_text:
        print(f"{name:<32} : {value_str} {unit_text}")
    else:
        print(f"{name:<32} : {value_str}")


def print_end(color="", reset="\033[0m"):
    line = "=" * 42

    if color:
        print(f"{color}{line}\n{reset}")
    else:
        print(f"{line}\n")