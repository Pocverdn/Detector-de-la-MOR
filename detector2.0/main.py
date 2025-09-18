import DetectionFaces as detect


def main():
    embs = detect.get_embs("photos")

    detect.visual_embs(embs)

if __name__ == "__main__":
    main()