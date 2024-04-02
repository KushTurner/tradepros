package main

import (
	"context"
	"encoding/base64"
	"log"
	"net/http"
	"os"

	firebase "firebase.google.com/go"
	"github.com/kushturner/tradepros/handlers"
	custommiddleware "github.com/kushturner/tradepros/middleware"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"google.golang.org/api/option"
)

func main() {
	keys := []string{os.Getenv("FINNHUB_KEY")}

	e := echo.New()
	v1 := e.Group("/api/v1")

	baseUrl := os.Getenv("BASE_URL")

	v1.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"https://www." + baseUrl, "https://" + baseUrl},
		AllowHeaders: []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept, echo.HeaderAuthorization},
		AllowMethods: []string{http.MethodGet, http.MethodHead, http.MethodPut, http.MethodPatch, http.MethodPost, http.MethodDelete},
	}))

	ctx := context.Background()
	fbKeyEncoded := os.Getenv("FIREBASE_KEY")
	sDec, err := base64.StdEncoding.DecodeString(fbKeyEncoded)
	if err != nil {
		log.Fatal(err)
	}
	sa := option.WithCredentialsJSON(sDec)
	app, err := firebase.NewApp(ctx, nil, sa)
	if err != nil {
		log.Fatalln(err)
	}

	client, err := app.Firestore(ctx)
	if err != nil {
		log.Fatalln(err)
	}

	defer client.Close()

	authClient, err := app.Auth(ctx)
	if err != nil {
		log.Fatalln(err)
	}

	jwtTokenMiddleware := custommiddleware.VerifyAuthMiddleware(ctx, authClient)
	finnhubKeyMiddleware := custommiddleware.RotateAPIKeysMiddleware(keys)

	v1.GET("/me", handlers.CurrentUserHandler(ctx, client), jwtTokenMiddleware)
	v1.GET("/stock/prediction", handlers.StockPredictionHandler())
	v1.GET("/stock", handlers.CompanyDataHandler(), finnhubKeyMiddleware)
	v1.GET("/stock/candle", handlers.HistoricalDataHandler(), finnhubKeyMiddleware)
	v1.GET("/search", handlers.SearchStockHandler())
	v1.GET("/leaderboard", handlers.LeaderboardHandler(ctx, client))
	v1.GET("/history", handlers.TradeHistoryHandler(ctx, client), jwtTokenMiddleware)
	v1.GET("/transactions", handlers.TransactionHandler(ctx, client), jwtTokenMiddleware)
	v1.GET("/transaction", handlers.SingleTransactionHandler(ctx, client), jwtTokenMiddleware)
	v1.GET("/watchlist", handlers.WatchlistHandler(ctx, client), jwtTokenMiddleware, finnhubKeyMiddleware)
	v1.POST("/register", handlers.RegisterUserHandler(ctx, client, authClient))
	v1.DELETE("/watchlist/:stock_id", handlers.RemoveWatchlistHandler(ctx, client), jwtTokenMiddleware)
	v1.GET("/watchlist/:stock_id", handlers.CheckWatchlistHandler(ctx, client), jwtTokenMiddleware)
	v1.POST("/watchlist", handlers.AddWatchlistHandler(ctx, client), jwtTokenMiddleware)
	v1.POST("/stock/buy", handlers.BuyStockHandler(ctx, client), jwtTokenMiddleware, finnhubKeyMiddleware)
	v1.POST("/stock/sell", handlers.SellStockHandler(ctx, client), jwtTokenMiddleware, finnhubKeyMiddleware)

	e.Logger.Fatal(e.Start(":1323"))
}
