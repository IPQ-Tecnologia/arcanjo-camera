from app.services.scene_analyzer import classify_rgb_color


CASES = {
    "pure black": ((18, 18, 18), "preta"),
    "black under warm light": ((55, 45, 35), "preta"),
    "dark blue": ((25, 42, 78), "azul-escura"),
    "blue": ((50, 100, 190), "azul"),
    "dark gray": ((90, 90, 90), "cinza-escura"),
    "gray": ((160, 160, 160), "cinza"),
    "white": ((235, 235, 235), "branca"),
    "brown": ((110, 70, 42), "marrom"),
    "red": ((190, 45, 45), "vermelha"),
    "green": ((55, 145, 70), "verde"),
    "yellow": ((220, 200, 45), "amarela"),
}


def run_test() -> None:
    print("===== COLOR CLASSIFICATION =====")

    for name, (rgb, expected) in CASES.items():
        result = classify_rgb_color(rgb)

        print(f"{name}: RGB={rgb} result={result}")

        assert result == expected, f"{name}: expected={expected}, got={result}"

    print("\nTest completed successfully.")


if __name__ == "__main__":
    run_test()
