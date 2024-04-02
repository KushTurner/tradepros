# build
FROM node:21-alpine3.18 as builder

WORKDIR /app

COPY package.json .

RUN npm install

COPY . .

RUN npm run build

FROM caddy:2.7.6-alpine
 
COPY Caddyfile /etc/caddy/Caddyfile

COPY --from=builder /app/dist /var/www/html


