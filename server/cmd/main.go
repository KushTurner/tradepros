package main

import (
	"log"

	"github.com/gofiber/fiber/v2"
	"github.com/kushturner/tradepros/api/db"
	"github.com/kushturner/tradepros/api/routes"
	_ "github.com/lib/pq"
)

func main() {

	db := db.SetupDB()

	defer db.Close()

	app := fiber.New()

	routes.SetupRoutes(app, db)

	app.Listen(":3000")

	log.Println(db)

}
