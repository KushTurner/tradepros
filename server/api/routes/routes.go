package routes

import (
	"database/sql"

	"github.com/gofiber/fiber/v2"
	"github.com/kushturner/tradepros/api/handlers"
)

func SetupRoutes(app *fiber.App, db *sql.DB) {
	app.Post("/api/register", handlers.RegisterAccountHandler(db))
}
