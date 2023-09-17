package custommiddleware

import (
	"context"
	"net/http"
	"strings"

	"firebase.google.com/go/auth"
	"github.com/labstack/echo/v4"
)

func VerifyAuthMiddleware(ctx context.Context, client *auth.Client) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			// Get the Authorization header from the request
			authHeader := c.Request().Header.Get("Authorization")

			// Check if the header is empty or doesn't start with "Bearer "
			if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
				return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorised")
			}

			// Extract the JWT token by removing the "Bearer " prefix
			idToken := strings.TrimPrefix(authHeader, "Bearer ")

			token, err := client.VerifyIDToken(ctx, idToken)
			if err != nil {
				return echo.NewHTTPError(http.StatusUnauthorized, "Unauthorised")
			}

			// Store token in request context
			c.Set("token", token)

			// Continue to the next handler in the chain
			return next(c)
		}
	}
}
