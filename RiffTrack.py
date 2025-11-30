import cv2
import time
import pygame
from modules import HandTrackingModule as ht
from modules.GameModule import GameModule, findSongPathing, getSongName


def main():
    minPathing = "./songs"
    audio = findSongPathing(minPathing)
    songName = getSongName(minPathing)
    game = GameModule(audio)

    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if not success:
        print("Could not access webcam.")
        return

    frame_height, frame_width, _ = frame.shape

    pygame.init()

    hud_height = 120
    window_size = (frame_width, frame_height + hud_height)
    screen = pygame.display.set_mode(window_size)

    pygame.display.set_caption("RiffTrack")

    try:
        icon = pygame.image.load("icon.png")
        pygame.display.set_icon(icon)
    except Exception:
        pass

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    detector = ht.HandDetector(detectionConfidence=0.65, trackingConfidence=0.5)

    volumes = [0, 0, 0, 0]
    pTime = 0
    running = True

    track_names = ["Guitar", "Vocals", "Drums", "Bass"]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False


        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList and not game.started:
            game.start()

        volumes = detector.detectVolume(img, lmList, volumes=volumes)

        game.changeVolume(volumes)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # === Convert OpenCV (BGR) frame to pygame surface (RGB) ===
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(
            img_rgb.tobytes(),
            (frame_width, frame_height),
            "RGB",
        )

        # Blit camera feed
        screen.blit(frame_surface, (0, 0))

        # === Draw HUD (volume bars) ===
        screen.fill((15, 15, 15), (0, frame_height, frame_width, hud_height))

        bar_width = 50
        max_bar_height = 80
        spacing = 40
        start_x = (bar_width * 4 + spacing * 4) / 2
        base_y = frame_height + hud_height - 10  # bottom of HUD

        songLabel = font.render(songName, True, (255, 255, 255))
        label_rect = songLabel.get_rect(
            center=(frame_width // 2, frame_height + hud_height // 2 - 70)
        )
        screen.blit(songLabel, label_rect)

        for i, vol in enumerate(volumes):
            # 0 – 100 -> bar height
            vol_clamped = max(0, min(100, vol))
            bar_height = int((vol_clamped / 100) * max_bar_height)

            x = start_x + i * (bar_width + spacing)
            y = base_y - bar_height

            # bar
            pygame.draw.rect(
                screen,
                (0, 200, 0),
                pygame.Rect(x, y, bar_width, bar_height),
            )

            # border
            pygame.draw.rect(
                screen,
                (200, 200, 200),
                pygame.Rect(x, base_y - max_bar_height, bar_width, max_bar_height),
                2,
            )

            # track name
            label = font.render(track_names[i], True, (255, 255, 255))
            label_rect = label.get_rect(center=(x + bar_width // 2, frame_height + 15))
            screen.blit(label, label_rect)

            # volume percent
            vol_text = font.render(f"{int(vol_clamped)}%", True, (200, 200, 200))
            vol_rect = vol_text.get_rect(center=(x + bar_width // 2, frame_height + 40))
            screen.blit(vol_text, vol_rect)

        # Update display
        pygame.display.flip()

        # Limit FPS (camera + pygame)
        clock.tick(60)

    # === Cleanup ===
    game.quit()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
