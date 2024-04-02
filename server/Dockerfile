FROM golang:1.22.1-alpine3.19 AS builder

WORKDIR /tmp/tradepros-api

COPY go.mod .

COPY go.sum .

RUN go mod download

COPY . .

RUN cd cmd && go build -o main .

FROM alpine:3.19

COPY --from=builder /tmp/tradepros-api/cmd/main /app/main

EXPOSE 1323

CMD ["/app/main"]