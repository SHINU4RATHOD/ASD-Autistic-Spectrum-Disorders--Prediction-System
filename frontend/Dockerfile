FROM node:18

WORKDIR /app

# Copy frontend files
COPY package.json yarn.lock ./
RUN yarn install

# Copy the rest of the frontend code
COPY . .

# Build the React app
RUN yarn build

# Install serve to run the production build
RUN yarn global add serve

# Expose port for the frontend
EXPOSE 3000

# Serve the built app
CMD ["serve", "-s", "build", "-l", "3000"]