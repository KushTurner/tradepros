FROM golang:1.20

WORKDIR /usr/src/app

RUN go install github.com/cosmtrek/air@latest

# Copy source code directories
COPY api .
COPY cmd .

# Copy development files
COPY .env .
COPY .air.toml .

# Copy Go module files
COPY go.mod .
COPY go.sum .



