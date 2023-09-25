package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"cloud.google.com/go/firestore"
	"firebase.google.com/go/auth"
	"github.com/kushturner/tradepros/models"
	"github.com/labstack/echo/v4"
	"google.golang.org/api/iterator"
)

func CurrentUserHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		// Retrieve the token from request context
		token, ok := c.Get("token").(*auth.Token)

		// Check if token is not of type *auth.Token or nil
		if !ok || token == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorised")
		}

		// Retrieve user ID from token
		userID := token.UID

		docRef := client.Collection("Users").Doc(userID)
		doc, err := docRef.Get(ctx)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get document")
		}

		// Return data in JSON format
		return c.JSON(http.StatusOK, doc.Data())
	}
}

func RegisterUserHandler(ctx context.Context, client *firestore.Client, authClient *auth.Client) echo.HandlerFunc {
	return func(c echo.Context) error {

		// Retrieve body

		var user models.UserRegistration
		if err := c.Bind(&user); err != nil {
			return c.JSON(http.StatusBadRequest, "Invalid request body")
		}

		// Body validaiton (username long enough, password long enough ect ect)

		if user.Username == "" || user.Email == "" || user.Password == "" || user.ConfirmPassword == "" {
			return echo.NewHTTPError(http.StatusBadRequest, "Username, email and password are required!")
		}

		// Username less than n

		if len(user.Username) < 6 {
			return echo.NewHTTPError(http.StatusBadRequest, "Username must be atleast 6 chars!")
		}

		// Password is less than n

		if len(user.Password) < 6 {
			return echo.NewHTTPError(http.StatusBadRequest, "Password must be atleast 6 chars!")
		}

		// Password not the same

		if user.Password != user.ConfirmPassword {
			return echo.NewHTTPError(http.StatusBadRequest, "Passwords do not match!")
		}

		// Check if user.Username is taken

		query := client.Collection("Users").Where("username", "==", user.Username)
		iter := query.Documents(ctx)
		_, err := iter.Next()
		if err != iterator.Done {
			return echo.NewHTTPError(http.StatusBadRequest, err)
		}

		// Create user

		params := (&auth.UserToCreate{}).
			Email(user.Email).
			EmailVerified(false).
			Password(user.Password).
			Disabled(false)
		u, err := authClient.CreateUser(ctx, params)

		if err != nil {
			if auth.IsEmailAlreadyExists(err) {
				return echo.NewHTTPError(http.StatusInternalServerError, "Email address already exists!")
			}
			if auth.IsInvalidEmail(err) {
				return echo.NewHTTPError(http.StatusInternalServerError, "Email address is invalid")
			}
		}

		// Initialize User document

		_, err = client.Collection("Users").Doc(u.UID).Set(ctx, map[string]interface{}{
			"balance":    100000,
			"created_at": firestore.ServerTimestamp,
			"username":   user.Username,
			"watchlist":  []interface{}{},
		})

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to initialize user!")
		}

		// Initialize Leaderboard document

		_, _, err = client.Collection("Leaderboard").Add(ctx, map[string]interface{}{
			"balance_gain": 0,
			"user_id":      u.UID,
		})

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to initialize user!")
		}

		// Generate custom token to log user in

		token, err := authClient.CustomToken(ctx, u.UID)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get token!")
		}

		// Send JWT in payload

		return c.String(http.StatusOK, token)
	}
}

func CompanyDataHandler() echo.HandlerFunc {
	return func(c echo.Context) error {
		url := os.Getenv("FINNHUB_URL")
		symbol := c.QueryParam("symbol")
		token := os.Getenv("FINNHUB_TOKEN")
		var stockData models.StockData

		profileEndpoint := fmt.Sprintf("%s/stock/profile2?symbol=%s&token=%s", url, symbol, token)
		financialsEndpoint := fmt.Sprintf("%s/stock/metric?symbol=%s&metric=all&token=%s", url, symbol, token)
		quoteEndpoint := fmt.Sprintf("%s/quote?symbol=%s&token=%s", url, symbol, token)

		// Profile2 Endpoint
		profile2, err := http.Get(profileEndpoint) // marketCapitalization, name, ticker, logo

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		defer profile2.Body.Close()
		json.NewDecoder(profile2.Body).Decode(&stockData)

		// Financials endpoint
		metric, err := http.Get(financialsEndpoint) // response.metric 52WeekHigh, 52WeekLow

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		defer metric.Body.Close()
		json.NewDecoder(metric.Body).Decode(&stockData)

		// Quote endpoint
		quote, err := http.Get(quoteEndpoint) // c

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		defer quote.Body.Close()
		json.NewDecoder(quote.Body).Decode(&stockData)

		return c.JSON(http.StatusOK, stockData)

	}
}

func StockPredictionHandler() echo.HandlerFunc {
	return func(c echo.Context) error {
		awsURL := os.Getenv("AWS_URL")
		awsKey := os.Getenv("AWS_TOKEN")
		todaysDate := time.Now().Format("2006-01-02")
		symbol := c.QueryParam("symbol")
		var stockPrediction models.StockPrediction

		// Stock prediction

		req, _ := http.NewRequest("GET", fmt.Sprintf("%s/tradepros?ticker=%s&date_to_predict=%s", awsURL, symbol, todaysDate), nil)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", awsKey)
		resp, err := http.DefaultClient.Do(req)

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		defer resp.Body.Close()

		json.NewDecoder(resp.Body).Decode(&stockPrediction)

		return c.JSON(http.StatusOK, stockPrediction)

	}
}

func HistoricalDataHandler() echo.HandlerFunc {
	return func(c echo.Context) error {
		url := os.Getenv("FINNHUB_URL")
		symbol := c.QueryParam("symbol")
		resolution := c.QueryParam("resolution")
		from := c.QueryParam("from")
		to := c.QueryParam("to")
		token := os.Getenv("FINNHUB_TOKEN")
		var historicalData models.HistoricalData

		endpoint := fmt.Sprintf("%s/stock/candle?symbol=%s&resolution=%s&from=%s&to=%s&token=%s", url, symbol, resolution, from, to, token)

		hist, err := http.Get(endpoint) // []c, []t, s

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		defer hist.Body.Close()
		json.NewDecoder(hist.Body).Decode(&historicalData)
		return c.JSON(http.StatusOK, historicalData)
	}
}

func SearchStockHandler() echo.HandlerFunc {
	// I add this during development/production as it's too many keys to add to .env
	keys := []string{}

	var keyIndex int

	return func(c echo.Context) error {
		key := keys[keyIndex]
		keyIndex = (keyIndex + 1) % len(keys)
		keywords := c.QueryParam("keywords")
		endpoint := fmt.Sprintf("https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=%s&apikey=%s", keywords, key)

		data, err := http.Get(endpoint)

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}

		var responseData map[string]interface{}
		json.NewDecoder(data.Body).Decode(&responseData)
		fmt.Println(key)
		return c.JSON(http.StatusOK, responseData)
	}
}
