# Custom Prometheus image whose scrape target is templated at start-up.
#
# ONE image for local, on-prem (Portainer) and cloud (ACA). The only thing that
# changes per environment is the BACKEND_TARGET / SCRAPE_SCHEME env vars read by
# docker-entrypoint.sh. A templated image is necessary because neither Portainer
# (stack content is pushed via API, with no repo files on the host) nor Azure
# Container Apps (no bind mounts) can mount a config file from the repo.
FROM prom/prometheus:v3.1.0

# Template + entrypoint sit in the default config dir, which the base image makes
# writable by the unprivileged 'nobody' user it runs as, so the rendered config
# can be written there at start-up without running as root.
COPY prometheus.tmpl.yml /etc/prometheus/prometheus.tmpl.yml
COPY docker-entrypoint.sh /etc/prometheus/docker-entrypoint.sh

# Invoke through `sh` so we do not depend on the executable bit surviving a
# checkout on Windows, where Git may not preserve file mode bits.
ENTRYPOINT ["/bin/sh", "/etc/prometheus/docker-entrypoint.sh"]
