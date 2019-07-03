module.exports = {
  apps : [{
    name: 'echo-python',
    cmd: 'app.py',
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
  }, {
    name: 'echo-python-3',
    cmd: 'app.py',
    interpreter: 'python3'
  }]
};

