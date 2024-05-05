package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
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
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		// Retrieve user ID from token
		userID := token.UID

		docRef := client.Collection("Users").Doc(userID)
		doc, err := docRef.Get(ctx)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get document")
		}

		leaderboardDoc := client.Collection("Leaderboard").Where("user_id", "==", userID)
		leaderboardRef, _ := leaderboardDoc.Documents(ctx).GetAll()
		currentUserStats := leaderboardRef[0]
		balance_gain, _ := currentUserStats.DataAt("balance_gain")

		response := doc.Data()
		response["balance_gain"] = balance_gain

		// Return data in JSON format
		return c.JSON(http.StatusOK, response)
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

		// Email validation

		if !isCommonEmailDomain(user.Email) {
			return echo.NewHTTPError(http.StatusBadRequest, "Invalid email!")
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
		token := c.Get("finnhubKey").(string)
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

		if stockData.Ticker == "" {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}

		return c.JSON(http.StatusOK, stockData)

	}
}

func StockPredictionHandler() echo.HandlerFunc {
	return func(c echo.Context) error {
		awsURL := os.Getenv("AWS_URL")
		awsKey := os.Getenv("AWS_KEY")
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
		url := os.Getenv("MARKETDATA_URL")
		symbol := c.QueryParam("symbol")
		resolution := c.QueryParam("resolution")
		from := c.QueryParam("from")
		to := c.QueryParam("to")
		token := os.Getenv("MARKETDATA_TOKEN")
		var historicalData models.HistoricalData

		endpoint := fmt.Sprintf("%s/stocks/candles/%s/%s?from=%s&to=%s", url, resolution, symbol, from, to)
		req, _ := http.NewRequest("GET", endpoint, nil)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		fmt.Println(req)

		resp, err := http.DefaultClient.Do(req) // []c, []t, s

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		defer resp.Body.Close()
		json.NewDecoder(resp.Body).Decode(&historicalData)
		return c.JSON(http.StatusOK, historicalData)
	}
}

func SearchStockHandler() echo.HandlerFunc {
	keys := []string{os.Getenv("ALPHAVANTAGE_KEY")}

	var keyIndex int

	return func(c echo.Context) error {
		key := keys[keyIndex]
		keyIndex = (keyIndex + 1) % len(keys)
		keywords := c.QueryParam("keywords")
		url := os.Getenv("ALPHAVANTAGE_URL")
		endpoint := fmt.Sprintf("%s/query?function=SYMBOL_SEARCH&keywords=%s&apikey=%s", url, keywords, key)

		data, err := http.Get(endpoint)

		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
		}
		var responseData map[string]interface{}
		json.NewDecoder(data.Body).Decode(&responseData)
		return c.JSON(http.StatusOK, responseData)
	}
}

func LeaderboardHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		var leaderboard []models.LeaderboardData
		var counter int64 = 1
		iter := client.Collection("Leaderboard").OrderBy("balance_gain", firestore.Desc).Limit(10).Documents(ctx)
		for {
			leaderboardDoc, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				return err
			}
			userID, _ := leaderboardDoc.DataAt("user_id")
			balance_gain, _ := leaderboardDoc.DataAt("balance_gain")
			userDoc, _ := client.Collection("Users").Doc(userID.(string)).Get(ctx)
			username, _ := userDoc.DataAt("username")

			leaderboardData := models.LeaderboardData{
				Rank:    counter,
				User:    username.(string),
				Balance: balance_gain,
			}
			leaderboard = append(leaderboard, leaderboardData)
			counter += 1
		}

		return c.JSON(http.StatusOK, leaderboard)
	}
}

func TradeHistoryHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		token, ok := c.Get("token").(*auth.Token)

		if !ok || token == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}
		var tradeHistory []models.TradeHistoryData

		iter := client.Collection("TradeHistory").Where("user_id", "==", token.UID).Documents(ctx)
		for {
			historyDoc, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				return err
			}
			symbol, _ := historyDoc.DataAt("stock_symbol")
			transactionPrice, _ := historyDoc.DataAt("transaction_price")
			units, _ := historyDoc.DataAt("units")
			transactionDate, _ := historyDoc.DataAt("transaction_date")
			date := transactionDate.(time.Time).Format("02/01/06")

			history := models.TradeHistoryData{
				Name:     symbol.(string),
				Action:   "sell",
				Amount:   transactionPrice,
				Quantity: units,
				Date:     date,
			}

			tradeHistory = append(tradeHistory, history)
		}

		return c.JSON(http.StatusOK, tradeHistory)

	}
}

func TransactionHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		token, ok := c.Get("token").(*auth.Token)

		if !ok || token == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}
		var transactions []models.TransactionData
		iter := client.Collection("Transactions").Where("user_id", "==", token.UID).Where("transaction_type", "==", "buy").Documents(ctx)
		for {
			transaction, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				return err
			}
			name, _ := transaction.DataAt("company_name")
			invested, _ := transaction.DataAt("transaction_price")
			units, _ := transaction.DataAt("units")

			history := models.TransactionData{
				Name:     name.(string),
				Quantity: units,
				Invested: invested,
			}

			transactions = append(transactions, history)
		}

		return c.JSON(http.StatusOK, transactions)

	}
}

func SingleTransactionHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		token, ok := c.Get("token").(*auth.Token)

		if !ok || token == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		symbol := c.QueryParam("symbol")

		transactionDoc, _ := client.Collection("Transactions").Where("user_id", "==", token.UID).Where("company_name", "==", symbol).Documents(ctx).GetAll()

		if len(transactionDoc) == 0 {
			return c.JSON(http.StatusOK, 0)
		}

		transaction, _ := transactionDoc[0].Ref.Get(ctx)

		price, _ := transaction.DataAt("transaction_price")

		return c.JSON(http.StatusOK, price)

	}
}

func WatchlistHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		url := os.Getenv("FINNHUB_URL")
		token := c.Get("finnhubKey").(string)
		var watchlistData []models.WatchlistData

		user, ok := c.Get("token").(*auth.Token)

		if !ok || user == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		query, _ := client.Collection("Users").Doc(user.UID).Get(ctx)
		watchlist, _ := query.DataAt("watchlist")

		for _, item := range watchlist.([]interface{}) {
			ticker := item.(string)
			profileEndpoint := fmt.Sprintf("%s/stock/profile2?symbol=%s&token=%s", url, ticker, token)
			profile2, err := http.Get(profileEndpoint)
			if err != nil {
				return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get response")
			}
			var temp models.WatchlistData
			json.NewDecoder(profile2.Body).Decode(&temp)
			watchlistData = append(watchlistData, temp)
			profile2.Body.Close()
		}
		return c.JSON(http.StatusOK, watchlistData)
	}
}

func RemoveWatchlistHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {

		stock_id := c.Param("stock_id")

		if stock_id == "" {
			return c.JSON(http.StatusBadRequest, "Missing parameter")
		}

		user, ok := c.Get("token").(*auth.Token)

		if !ok || user == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		docRef := client.Collection("Users").Doc(user.UID)

		watchlistContext, _ := docRef.Get(ctx)
		watchlist, _ := watchlistContext.DataAt("watchlist")

		var newWatchlist []string
		for _, item := range watchlist.([]interface{}) {
			if item != stock_id {
				newWatchlist = append(newWatchlist, item.(string))
			}
		}

		if len(newWatchlist) == 0 {
			docRef.Set(ctx, map[string]interface{}{
				"watchlist": []string{},
			}, firestore.MergeAll)
		} else {
			docRef.Set(ctx, map[string]interface{}{
				"watchlist": newWatchlist,
			}, firestore.MergeAll)
		}

		return c.JSON(http.StatusOK, "Updated")
	}
}

func AddWatchlistHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {

		stock_id := c.FormValue("stock")

		if stock_id == "" {
			return c.JSON(http.StatusBadRequest, "Missing parameter")
		}

		user, ok := c.Get("token").(*auth.Token)

		if !ok || user == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		docRef := client.Collection("Users").Doc(user.UID)

		watchlistContext, _ := docRef.Get(ctx)
		watchlist, _ := watchlistContext.DataAt("watchlist")

		newWatchlist := watchlist.([]interface{})

		for _, item := range newWatchlist {
			if item == stock_id {
				return c.JSON(http.StatusBadRequest, "Stock is already in the watchlist")
			}

		}

		newWatchlist = append(newWatchlist, stock_id)

		docRef.Set(ctx, map[string]interface{}{
			"watchlist": newWatchlist,
		}, firestore.MergeAll)

		return c.JSON(http.StatusOK, "Updated")
	}
}

func CheckWatchlistHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		user, ok := c.Get("token").(*auth.Token)

		if !ok || user == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		stock_id := c.Param("stock_id")

		if stock_id == "" {
			return c.JSON(http.StatusBadRequest, "Missing parameter")
		}

		docRef := client.Collection("Users").Doc(user.UID)

		watchlistContext, _ := docRef.Get(ctx)
		watchlist, _ := watchlistContext.DataAt("watchlist")

		newWatchlist := watchlist.([]interface{})

		for _, item := range newWatchlist {
			if item == stock_id {
				return c.JSON(http.StatusOK, true)
			}
		}
		return c.JSON(http.StatusOK, false)
	}
}

func BuyStockHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		user, ok := c.Get("token").(*auth.Token)
		url := os.Getenv("FINNHUB_URL")

		if !ok || user == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		token := c.Get("finnhubKey").(string)
		var purchase models.PurchaseData

		c.Bind(&purchase)

		if purchase.Stock == "" || purchase.Price == 0.0 {
			return echo.NewHTTPError(http.StatusBadRequest, "Missing values")
		}

		endpoint := fmt.Sprintf("%s/quote?symbol=%s&token=%s", url, purchase.Stock, token)
		req, err := http.Get(endpoint)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Invalid Stock")
		}
		defer req.Body.Close()

		var response models.Response
		json.NewDecoder(req.Body).Decode(&response)
		value := purchase.Price / response.C
		units := value
		price := units * response.C

		profileDocRef := client.Collection("Users").Doc(user.UID)

		userProfile, _ := profileDocRef.Get(ctx)
		balance, _ := userProfile.DataAt("balance")

		var balanceFloat64 float64

		if balanceValue, ok := balance.(float64); ok {
			balanceFloat64 = balanceValue
		} else if balanceValue, ok := balance.(int64); ok {
			balanceFloat64 = float64(balanceValue)
		}

		if balanceFloat64 < price {
			return echo.NewHTTPError(http.StatusBadRequest, "Insufficient Funds")
		}

		newBalance := balanceFloat64 - price

		profileDocRef.Set(ctx, map[string]interface{}{
			"balance": newBalance,
		}, firestore.MergeAll)

		transactionDocRef := client.Collection("Transactions")

		transactionQuery := transactionDocRef.
			Where("transaction_type", "==", "buy").
			Where("user_id", "==", user.UID).
			Where("company_name", "==", purchase.Stock)

		querySnapshot, _ := transactionQuery.Documents(ctx).GetAll()

		if len(querySnapshot) == 1 {
			matchDocRef := querySnapshot[0].Ref
			originalPriceRef, _ := matchDocRef.Get(ctx)
			originalUnitsRef, _ := matchDocRef.Get(ctx)
			originalPrice, _ := originalPriceRef.DataAt("transaction_price")
			originalUnits, _ := originalUnitsRef.DataAt("units")
			var originalPriceFloat64 float64
			var originalUnitsFloat64 float64

			if priceValue, ok := originalPrice.(float64); ok {
				originalPriceFloat64 = priceValue
			} else if priceValue, ok := originalPrice.(int64); ok {
				originalPriceFloat64 = float64(priceValue)
			}

			if unitValue, ok := originalUnits.(float64); ok {
				originalUnitsFloat64 = unitValue
			} else if unitValue, ok := originalUnits.(int64); ok {
				originalUnitsFloat64 = float64(unitValue)
			}

			matchDocRef.Set(ctx, map[string]interface{}{
				"company_name":      purchase.Stock,
				"transaction_price": originalPriceFloat64 + price,
				"transaction_type":  "buy",
				"units":             originalUnitsFloat64 + units,
				"user_id":           user.UID,
			}, firestore.MergeAll)

			return c.JSON(http.StatusOK, "Purchased Merged")
		}

		transactionDocRef.Add(ctx, map[string]interface{}{
			"company_name":      purchase.Stock,
			"transaction_price": price,
			"transaction_type":  "buy",
			"units":             units,
			"user_id":           user.UID,
		})

		return c.JSON(http.StatusOK, "Purchased")
	}
}

func SellStockHandler(ctx context.Context, client *firestore.Client) echo.HandlerFunc {
	return func(c echo.Context) error {
		user, ok := c.Get("token").(*auth.Token)
		url := os.Getenv("FINNHUB_URL")
		token := c.Get("finnhubKey").(string)

		if !ok || user == nil {
			return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorized")
		}

		var purchase models.PurchaseData

		c.Bind(&purchase)

		endpoint := fmt.Sprintf("%s/quote?symbol=%s&token=%s", url, purchase.Stock, token)
		req, err := http.Get(endpoint)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, "Invalid Stock")
		}
		defer req.Body.Close()

		var response models.Response
		json.NewDecoder(req.Body).Decode(&response)

		profileDocRef := client.Collection("Users").Doc(user.UID)

		userProfile, _ := profileDocRef.Get(ctx)
		balance, _ := userProfile.DataAt("balance")

		var balanceFloat64 float64

		if balanceValue, ok := balance.(float64); ok {
			balanceFloat64 = balanceValue
		} else if balanceValue, ok := balance.(int64); ok {
			balanceFloat64 = float64(balanceValue)
		}

		transactionDocRef := client.Collection("Transactions")

		transactionQuery := transactionDocRef.
			Where("transaction_type", "==", "buy").
			Where("user_id", "==", user.UID).
			Where("company_name", "==", purchase.Stock)

		querySnapshot, _ := transactionQuery.Documents(ctx).GetAll()

		if len(querySnapshot) == 0 {
			return echo.NewHTTPError(http.StatusBadRequest, "Insufficient Funds")
		}

		matchDocRef := querySnapshot[0].Ref
		originalRef, _ := matchDocRef.Get(ctx)
		originalPrice, _ := originalRef.DataAt("transaction_price")
		originalUnits, _ := originalRef.DataAt("units")
		var originalPriceFloat64 float64
		var originalUnitsFloat64 float64

		if priceValue, ok := originalPrice.(float64); ok {
			originalPriceFloat64 = priceValue
		} else if priceValue, ok := originalPrice.(int64); ok {
			originalPriceFloat64 = float64(priceValue)
		}

		if unitValue, ok := originalUnits.(float64); ok {
			originalUnitsFloat64 = unitValue
		} else if unitValue, ok := originalUnits.(int64); ok {
			originalUnitsFloat64 = float64(unitValue)
		}

		if originalPriceFloat64 < purchase.Price {
			return echo.NewHTTPError(http.StatusBadRequest, "Insufficient Funds")
		}

		if originalPriceFloat64 == purchase.Price {
			matchDocRef.Delete(ctx)
		}
		pricePerUnit := originalPriceFloat64 / originalUnitsFloat64
		percentIncrease := ((response.C - pricePerUnit) / pricePerUnit)
		gain := purchase.Price * (1 + percentIncrease)
		numberOfUnitsSold := purchase.Price / pricePerUnit

		if originalPriceFloat64 > purchase.Price {

			matchDocRef.Set(ctx, map[string]interface{}{
				"company_name":      purchase.Stock,
				"transaction_price": originalPriceFloat64 - purchase.Price,
				"transaction_type":  "buy",
				"units":             originalUnitsFloat64 - numberOfUnitsSold,
				"user_id":           user.UID,
			}, firestore.MergeAll)
		}

		profileDocRef.Set(ctx, map[string]interface{}{
			"balance": balanceFloat64 + gain,
		}, firestore.MergeAll)

		client.Collection("TradeHistory").Add(ctx, map[string]interface{}{
			"stock_symbol":      purchase.Stock,
			"transaction_price": purchase.Price,
			"transaction_date":  firestore.ServerTimestamp,
			"transaction_type":  "sell",
			"units":             numberOfUnitsSold,
			"user_id":           user.UID,
		})

		leaderboardDoc, _ := client.Collection("Leaderboard").Where("user_id", "==", user.UID).Documents(ctx).GetAll()
		leaderboardRef := leaderboardDoc[0].Ref
		leaderboardRefInfo, _ := leaderboardRef.Get(ctx)
		balanceGain, _ := leaderboardRefInfo.DataAt("balance_gain")

		var balanceGainFloat64 float64

		if balanceGainValue, ok := balanceGain.(float64); ok {
			balanceGainFloat64 = balanceGainValue
		} else if balanceGainValue, ok := balanceGain.(int64); ok {
			balanceGainFloat64 = float64(balanceGainValue)
		}

		leaderboardRef.Set(ctx, map[string]interface{}{
			"balance_gain": balanceGainFloat64 + (gain - purchase.Price),
		}, firestore.MergeAll)

		return c.JSON(http.StatusOK, "Sold")
	}
}

func isCommonEmailDomain(email string) bool {
	commonDomains := []string{"gmail.com", "hotmail.com", "yahoo.co.uk", "outlook.com", "yahoo.com", "icloud.com"}

	parts := strings.Split(email, "@")
	if len(parts) != 2 {
		return false
	}

	domain := parts[1]
	for _, commonDomain := range commonDomains {
		if strings.ToLower(domain) == commonDomain {
			return true
		}
	}

	return false
}
