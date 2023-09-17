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
