# FROM nginx:alpine
# COPY index.html /usr/share/nginx/html/index.html

# Use a lean Nginx image
FROM nginx:alpine

# Copy all your frontend static files into the Nginx public directory
# The '.' refers to the current directory where the Dockerfile is located (i.e., 'frontend/')
# The '/usr/share/nginx/html' is the default Nginx web root.
COPY . /usr/share/nginx/html

# Expose port 80, which is the default port Nginx listens on
EXPOSE 80

# Command to start Nginx in the foreground so Docker can keep the container running
CMD ["nginx", "-g", "daemon off;"]