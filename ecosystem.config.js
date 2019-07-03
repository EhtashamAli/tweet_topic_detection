module.exports = {
  apps : [{
    name: 'App',
    cmd: 'app.py',
    interpreter: 'python3',
    autorestart: false,
    watch: true,
    instances: 1,
    max_memory_restart: '1G',
    env: {
      ENV: 'development'
    },
    env_production : {
      ENV: 'production'
    }
  }]
};

