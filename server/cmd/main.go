package main

import (
	"context"
	"log"
	"net/http"

	firebase "firebase.google.com/go"
	"github.com/joho/godotenv"
	"github.com/kushturner/tradepros/handlers"
	custommiddleware "github.com/kushturner/tradepros/middleware"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"google.golang.org/api/option"
)

func main() {
	// Set up Echo
	e := echo.New()

	// Set up CORS middleware

	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"*"}, // "https://www.tradepros.live", "https://tradepros.live"
		AllowHeaders: []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept},
		AllowMethods: []string{http.MethodGet, http.MethodHead, http.MethodPut, http.MethodPatch, http.MethodPost, http.MethodDelete},
	}))

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// Set up Firestore
	ctx := context.Background()
	sa := option.WithCredentialsFile("../serviceAccount.json")
	app, err := firebase.NewApp(ctx, nil, sa)
	if err != nil {
		log.Fatalln(err)
	}

	client, err := app.Firestore(ctx)
	if err != nil {
		log.Fatalln(err)
	}

	defer client.Close()

	// Set up Firebase
	authClient, err := app.Auth(ctx)
	if err != nil {
		log.Fatalln(err)
	}

	// Routes

	e.GET("/me", handlers.CurrentUserHandler(ctx, client), custommiddleware.VerifyAuthMiddleware(ctx, authClient))
	e.GET("/stock", handlers.CompanyDataHandler())                //, custommiddleware.VerifyAuthMiddleware(ctx, authClient)
	e.GET("/stock/prediction", handlers.StockPredictionHandler()) //, custommiddleware.VerifyAuthMiddleware(ctx, authClient)
	e.GET("/stock/candle", handlers.HistoricalDataHandler())      //, custommiddleware.VerifyAuthMiddleware(ctx, authClient)
	e.POST("/register", handlers.RegisterUserHandler(ctx, client, authClient))

	// Serve Server

	e.Logger.Fatal(e.Start(":8080"))
}
