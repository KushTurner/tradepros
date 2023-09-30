package models

type UserRegistration struct {
	Username        string `json:"username"`
	Email           string `json:"email"`
	Password        string `json:"password"`
	ConfirmPassword string `json:"confirmPassword"`
}

type StockData struct {
	CompanyName   string  `json:"name"`
	Ticker        string  `json:"ticker"`
	Logo          string  `json:"logo"`
	MarketCap     float64 `json:"marketCapitalization"`
	Price         float64 `json:"c"`
	PreviousClose float64 `json:"pc"`
	Data          struct {
		WeekHigh float64 `json:"52WeekHigh"`
		WeekLow  float64 `json:"52WeekLow"`
	} `json:"metric"`
}

type StockPrediction struct {
	Confidence float64 `json:"confidence"`
	Direction  string  `json:"model_answer"`
}

type HistoricalData struct {
	C []float64 `json:"c"`
	T []int64   `json:"t"`
	S string    `json:"s"`
}

type LeaderboardData struct {
	Rank    int64       `json:"rank"`
	User    string      `json:"username"`
	Balance interface{} `json:"balance"`
}

type TradeHistoryData struct {
	Name     string      `json:"name"`
	Action   string      `json:"action"`
	Amount   interface{} `json:"amount"`
	Quantity interface{} `json:"quantity"`
	Date     string      `json:"date"`
}

type TransactionData struct {
	Name     string      `json:"name"`
	Quantity interface{} `json:"shares"`
	Invested interface{} `json:"investment"`
}

type WatchlistData struct {
	Ticker   string `json:"ticker"`
	Name     string `json:"name"`
	Image    string `json:"logo"`
	Industry string `json:"finnhubIndustry"`
}

type PurchaseData struct {
	Price float64 `json:"price"`
	Stock string  `json:"stock"`
}

type Response struct {
	C float64 `json:"c"`
}
