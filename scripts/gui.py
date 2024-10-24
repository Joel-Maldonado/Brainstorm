import pygame
import chess
import chess.engine


ENGINE_PATH = (
    "./brainstorm"  # Expecting 'brainstorm' executable must be in root directory!
)


class ChessGUI:
    def __init__(self):
        pygame.init()
        self.SQUARE_SIZE = 100
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        self.screen = pygame.display.set_mode((self.BOARD_SIZE, self.BOARD_SIZE))
        pygame.display.set_caption("Chess Game vs Model")

        self.board = chess.Board()

        # Load piece images
        self.pieces = {}
        pieces = ["p", "n", "b", "r", "q", "k"]
        for piece in pieces:
            # Load white pieces
            self.pieces[piece.upper()] = pygame.transform.smoothscale(
                pygame.image.load(f"./assets/pieces/w_{piece}.png").convert_alpha(),
                (self.SQUARE_SIZE, self.SQUARE_SIZE),
            )

            # Load black pieces
            self.pieces[piece] = pygame.transform.smoothscale(
                pygame.image.load(f"./assets/pieces/b_{piece}.png").convert_alpha(),
                (self.SQUARE_SIZE, self.SQUARE_SIZE),
            )

        self.selected_square = None
        self.valid_moves = []

        engine_path = ENGINE_PATH  # Adjust path as needed
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        # Colors
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT = (130, 151, 105)
        self.SELECTED = (186, 202, 43)

    def get_square_from_coords(self, pos):
        """Convert screen coordinates to chess square."""
        x, y = pos
        file = x // self.SQUARE_SIZE
        rank = 7 - (y // self.SQUARE_SIZE)
        return chess.square(file, rank)

    def get_coords_from_square(self, square):
        """Convert chess square to screen coordinates."""
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        return (file * self.SQUARE_SIZE, rank * self.SQUARE_SIZE)

    def draw_board(self):
        """Draw the chess board."""
        for rank in range(8):
            for file in range(8):
                color = (
                    self.LIGHT_SQUARE if (rank + file) % 2 == 0 else self.DARK_SQUARE
                )
                rect = pygame.Rect(
                    file * self.SQUARE_SIZE,
                    rank * self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                )
                pygame.draw.rect(self.screen, color, rect)

    def draw_pieces(self):
        """Draw the chess pieces."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = self.get_coords_from_square(square)
                piece_img = self.pieces[piece.symbol()]
                self.screen.blit(piece_img, (x, y))

    def highlight_squares(self):
        """Highlight selected square and valid moves."""
        if self.selected_square is not None:
            # Highlight selected square
            x, y = self.get_coords_from_square(self.selected_square)
            rect = pygame.Rect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
            pygame.draw.rect(self.screen, self.SELECTED, rect, 4)

            # Highlight valid moves
            for move in self.valid_moves:
                x, y = self.get_coords_from_square(move.to_square)
                rect = pygame.Rect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, self.HIGHLIGHT, rect, 4)

    def get_computer_move(self):
        """Get the best move from the chess engine."""
        result = self.engine.play(self.board, chess.engine.Limit(time=2.0))
        return result.move

    def update_display(self):
        """Update the game display."""
        self.draw_board()
        self.highlight_squares()
        self.draw_pieces()
        pygame.display.flip()

    def run(self):
        """Main game loop."""
        running = True
        player_turn = True  # True for white (player), False for black (computer)

        while running:
            if not player_turn:
                # Computer's turn
                move = self.get_computer_move()
                self.board.push(move)
                player_turn = True
                self.update_display()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and player_turn:
                    pos = pygame.mouse.get_pos()
                    square = self.get_square_from_coords(pos)

                    if self.selected_square is None:
                        # Select piece
                        piece = self.board.piece_at(square)
                        if piece and piece.color == chess.WHITE:
                            self.selected_square = square
                            self.valid_moves = [
                                move
                                for move in self.board.legal_moves
                                if move.from_square == square
                            ]
                    else:
                        # Try to make a move
                        move = chess.Move(self.selected_square, square)
                        if move in self.valid_moves:
                            self.board.push(move)
                            player_turn = False

                        # Reset selection
                        self.selected_square = None
                        self.valid_moves = []

            self.update_display()

            if self.board.is_game_over():
                result = (
                    "1-0"
                    if self.board.is_checkmate() and not self.board.turn
                    else "0-1"
                )
                print(f"Game Over! Result: {result}")
                running = False

        # Cleanup
        self.engine.quit()
        pygame.quit()


if __name__ == "__main__":
    game = ChessGUI()
    game.run()
