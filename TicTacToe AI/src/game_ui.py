import pygame
import torch
import pygame.gfxdraw

from environment import TicTacToeEnv
from dqn import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH = 600
HEIGHT = 750

BOARD_SIZE = 560
CELL = BOARD_SIZE // 3

TOP_MARGIN = 140

BG = (12, 25, 60)         
BOARD_BG = (12, 25, 70) 

CELL_BG = (242, 244, 247)
CELL_BORDER = (220, 225, 232)
GRID = (205, 210, 220)
TEXT = (255, 255, 255)

BUTTON = (95, 100, 205)
BUTTON_HOVER = (160, 145, 255)

SHADOW = (0, 0, 0)


def normalize_state(state, player):
    state = state.tolist()

    if player == 1:
        return state

    return [-x for x in state]


class Button:

    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, screen, font):

        mouse = pygame.mouse.get_pos()
        color = BUTTON

        if self.rect.collidepoint(mouse):
            color = BUTTON_HOVER

        pygame.draw.rect(screen, (20, 20, 40), self.rect.move(4, 4), border_radius=16)
        pygame.draw.rect(screen, color, self.rect, border_radius=16)
        txt = font.render(self.text, True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class GameUI:

    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("TicTacToe AI")

        self.clock = pygame.time.Clock()
        self.env = TicTacToeEnv()
        self.title_font = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.text_font = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 22)

        self.running = True
        self.state = "menu"
        self.player_turn = True
        self.game_over = False
        self.result_text = ""
        self.human_player = 1
        self.ai_player = -1
        self.model = None

        self.play_x_button = Button((150, 320, 300, 70),"Play as X")
        self.play_o_button = Button((150, 420, 300, 70), "Play as O")
        self.restart_button = Button((150, 500, 300, 70), "Play Again")

    def load_game(self, human_player):

        self.env.reset()
        self.game_over = False
        self.result_text = ""
        self.human_player = human_player
        self.ai_player = -human_player

        if human_player == 1:
            model_path = "models2/player2/best_model.pth"
            self.player_turn = True

        else:
            model_path = "models2/player1/best_model.pth"
            self.player_turn = False

        self.model = DQN().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path,map_location=DEVICE))
        self.model.eval()
        self.state = "game"

    def draw_menu(self):

        self.screen.fill(BG)
        title = self.title_font.render("TicTacToe AI", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, 160)))
        subtitle = self.text_font.render("Choose your side", True, TEXT)
        self.screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, 230)))
        self.play_x_button.draw(self.screen, self.text_font)
        self.play_o_button.draw( self.screen, self.text_font)

    def draw_board(self):

        board_rect = pygame.Rect(20, TOP_MARGIN, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, SHADOW, board_rect.move(6, 6), border_radius=24)
        pygame.draw.rect(self.screen, BOARD_BG, board_rect, border_radius=24)

        for r in range(3):
            for c in range(3):

                x = 20 + c * CELL
                y = TOP_MARGIN + r * CELL

                cell = pygame.Rect(x + 4, y + 4, CELL - 8, CELL - 8)
                shadow = cell.move(3, 3)
                pygame.draw.rect(self.screen, (150, 155, 165), shadow, border_radius=12)
                pygame.draw.rect( self.screen, CELL_BG, cell, border_radius=18)
                pygame.draw.rect(self.screen, CELL_BORDER, cell, 2, border_radius=18)


    def draw_symbols(self):

        board = self.env.board

        for r in range(3):
            for c in range(3):

                value = board[r][c]
                x = 20 + c * CELL
                y = TOP_MARGIN + r * CELL

                if value == 1:
                    
                    pad = CELL // 4
                    pygame.draw.line(
                        self.screen,
                        (100, 110, 200),
                        (x + pad, y + pad),
                        (x + CELL - pad, y + CELL - pad),
                        14
                    )

                    pygame.draw.line(
                        self.screen,
                        (80, 170, 255),
                        (x + CELL - pad, y + pad),
                        (x + pad, y + CELL - pad),
                        14
                    )

                elif value == -1:

                    center = (x + CELL // 2, y + CELL // 2)
                    radius = CELL // 3

                    pygame.gfxdraw.aacircle(
                        self.screen,
                        center[0],
                        center[1],
                        radius,
                        (220, 60, 60)
                    )

                    pygame.gfxdraw.aacircle(
                        self.screen,
                        center[0],
                        center[1],
                        radius - 1,
                        (220, 60, 60)
                    )

                    pygame.draw.circle(
                        self.screen,
                        (220, 60, 60),
                        center,
                        radius,
                        10
                    )

    def draw_status(self):

        if self.game_over:
            text = self.result_text

        elif self.player_turn:
            text = "Your Turn"

        else:
            text = "AI Thinking..."

        label = self.text_font.render(text, True, TEXT)
        self.screen.blit(label, label.get_rect(center=(WIDTH // 2, 60)))


    def draw_game_over_overlay(self):

        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        box = pygame.Rect(75, 220, 450, 150)

        pygame.draw.rect(
            self.screen,
            (20, 40, 90),
            box,
            border_radius=24
        )

        title = self.title_font.render(
            self.result_text,
            True,
            (255, 255, 255)
        )

        self.screen.blit(
            title,
            title.get_rect(center=(WIDTH // 2, 290))
        )

        self.restart_button.draw(
            self.screen,
            self.text_font
        )

    def ai_move(self):

        raw_state = self.env.get_state()
        valid = self.env.available_actions()

        if not valid:
            return

        state = normalize_state(raw_state, self.ai_player)

        with torch.no_grad():

            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.model(s)[0]

        masked = torch.full((9,), float("-inf"), device=DEVICE)

        for action in valid:
            masked[action] = q_values[action]

        move = torch.argmax(masked).item()

        _, _, done = self.env.step(
            move,
            player=self.ai_player
        )

        if done:

            winner = self.env.check_winner()

            if winner == self.ai_player:
                self.result_text = "AI Wins"

            elif winner == self.human_player:
                self.result_text = "You Win"

            else:
                self.result_text = "Draw"

            self.game_over = True

        self.player_turn = True

    def handle_human_click(self, pos):

        if self.game_over:
            return

        x, y = pos

        if y < TOP_MARGIN:
            return

        board_x = x - 20
        board_y = y - TOP_MARGIN

        if board_x < 0 or board_y < 0:
            return

        col = board_x // CELL
        row = board_y // CELL

        if row > 2 or col > 2:
            return

        action = row * 3 + col

        if action not in self.env.available_actions():
            return

        _, _, done = self.env.step(
            action,
            player=self.human_player
        )

        if done:

            winner = self.env.check_winner()

            if winner == self.human_player:
                self.result_text = "You Win"

            elif winner == self.ai_player:
                self.result_text = "AI Wins"

            else:
                self.result_text = "Draw"

            self.game_over = True

        else:

            self.player_turn = False

    def draw_game(self):

        self.screen.fill(BG)

        self.draw_status()

        self.draw_board()

        self.draw_symbols()

        if self.game_over:
            self.draw_game_over_overlay()

    def handle_menu_click(self, pos):

        if self.play_x_button.clicked(pos):
            self.load_game(1)

        elif self.play_o_button.clicked(pos):
            self.load_game(-1)

    def restart(self):

        self.state = "menu"

        self.game_over = False

        self.result_text = ""

        self.env.reset()

    def run(self):

        while self.running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if self.state == "menu":
                        self.handle_menu_click(event.pos)

                    elif self.state == "game":

                        if self.game_over:

                            if self.restart_button.clicked(event.pos):
                                self.restart()

                        elif self.player_turn:

                            self.handle_human_click(event.pos)

            if (self.state == "game" and not self.game_over and not self.player_turn):
                self.ai_move()

            if self.state == "menu":
                self.draw_menu()

            elif self.state == "game":
                self.draw_game()

            pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    GameUI().run()