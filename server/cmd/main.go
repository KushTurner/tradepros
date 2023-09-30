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
	keys := []string{}

	// finnhubkeys

	// Set up Echo
	e := echo.New()

	// Set up CORS middleware

	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"*"}, // "https://www.tradepros.live", "https://tradepros.live"
		AllowHeaders: []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept, echo.HeaderAuthorization},
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

	// Define Middleware

	jwtTokenMiddleware := custommiddleware.VerifyAuthMiddleware(ctx, authClient)
	finnhubKeyMiddleware := custommiddleware.RotateAPIKeysMiddleware(keys)

	// Routes

	e.GET("/me", handlers.CurrentUserHandler(ctx, client), jwtTokenMiddleware)
	e.GET("/stock", handlers.CompanyDataHandler(), finnhubKeyMiddleware)
	e.GET("/stock/prediction", handlers.StockPredictionHandler())
	e.GET("/stock/candle", handlers.HistoricalDataHandler(), finnhubKeyMiddleware)
	e.GET("/search", handlers.SearchStockHandler())
	e.GET("/leaderboard", handlers.LeaderboardHandler(ctx, client))
	e.GET("/history", handlers.TradeHistoryHandler(ctx, client), jwtTokenMiddleware)
	e.GET("/transactions", handlers.TransactionHandler(ctx, client), jwtTokenMiddleware)
	e.GET("/transaction", handlers.SingleTransactionHandler(ctx, client), jwtTokenMiddleware)
	e.GET("/watchlist", handlers.WatchlistHandler(ctx, client), jwtTokenMiddleware, finnhubKeyMiddleware)
	e.POST("/register", handlers.RegisterUserHandler(ctx, client, authClient))
	e.DELETE("/watchlist/:stock_id", handlers.RemoveWatchlistHandler(ctx, client), jwtTokenMiddleware)
	e.GET("/watchlist/:stock_id", handlers.CheckWatchlistHandler(ctx, client), jwtTokenMiddleware)
	e.POST("/watchlist", handlers.AddWatchlistHandler(ctx, client), jwtTokenMiddleware)
	e.POST("/stock/buy", handlers.BuyStockHandler(ctx, client), jwtTokenMiddleware, finnhubKeyMiddleware)
	e.POST("/stock/sell", handlers.SellStockHandler(ctx, client), jwtTokenMiddleware, finnhubKeyMiddleware)

	// Serve Server

	e.Logger.Fatal(e.Start(":1323"))
}
