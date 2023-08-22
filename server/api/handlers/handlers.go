package handlers

import (
	"database/sql"

	"github.com/gofiber/fiber/v2"
	"github.com/kushturner/tradepros/api/models"
)

func RegisterAccountHandler(db *sql.DB) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var acc models.Person

		if err := c.BodyParser(&acc); err != nil {
			return err
		}

		_, err := db.Exec("INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3)",
			acc.Username, acc.Email, acc.Password)

		if err != nil {
			return err
		}

		return c.JSON(fiber.Map{"message": "User registered successfully"})
	}
}
