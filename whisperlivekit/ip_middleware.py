"""
IP restriction middleware for FastAPI
"""
import ipaddress
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class IPRestrictionMiddleware:
    """
    Middleware to restrict access based on IP ranges.

    Supports both IPv4 and IPv6 addresses and networks.
    """

    def __init__(self, allowed_ips=None, allowed_networks=None):
        """
        Initialize the IP restriction middleware.

        Args:
            allowed_ips: List of allowed IP addresses (strings)
            allowed_networks: List of allowed IP networks in CIDR notation (strings)
        """
        self.allowed_ips = set()
        self.allowed_networks = []

        # Process allowed IPs
        if allowed_ips:
            for ip_str in allowed_ips:
                try:
                    ip_obj = ipaddress.ip_address(ip_str.strip())
                    self.allowed_ips.add(ip_obj)
                except ValueError as e:
                    logger.warning(f"Invalid IP address: {ip_str} - {e}")

        # Process allowed networks
        if allowed_networks:
            for network_str in allowed_networks:
                try:
                    network_obj = ipaddress.ip_network(network_str.strip(), strict=False)
                    self.allowed_networks.append(network_obj)
                except ValueError as e:
                    logger.warning(f"Invalid IP network: {network_str} - {e}")

        # If no restrictions are set, allow all (for backward compatibility)
        self.has_restrictions = bool(self.allowed_ips or self.allowed_networks)

        if self.has_restrictions:
            logger.info(f"IP restrictions enabled. Allowed IPs: {len(self.allowed_ips)}, Networks: {len(self.allowed_networks)}")
        else:
            logger.warning("No IP restrictions configured - allowing all connections")

    def is_ip_allowed(self, client_ip):
        """
        Check if a client IP is allowed.

        Args:
            client_ip: IP address string

        Returns:
            bool: True if allowed, False otherwise
        """
        if not self.has_restrictions:
            return True

        try:
            ip_obj = ipaddress.ip_address(client_ip)

            # Check exact IP matches
            if ip_obj in self.allowed_ips:
                return True

            # Check network matches
            for network in self.allowed_networks:
                if ip_obj in network:
                    return True

            return False

        except ValueError:
            logger.warning(f"Invalid client IP address: {client_ip}")
            return False

    async def __call__(self, request: Request, call_next):
        """
        Middleware call function.
        """
        # Get client IP - handle both direct connections and proxies
        client_ip = None

        # Check X-Forwarded-For header (for proxies)
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            # Take the first IP in case of multiple proxies
            client_ip = x_forwarded_for.split(",")[0].strip()
        else:
            # Direct connection - use request.client.host
            client_ip = request.client.host if request.client else None

        if not client_ip:
            logger.warning("Could not determine client IP address")
            return JSONResponse(
                status_code=403,
                content={"error": "Could not determine client IP address"}
            )

        if not self.is_ip_allowed(client_ip):
            logger.warning(f"Access denied for IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": f"Access denied for IP: {client_ip}"}
            )

        # IP is allowed, proceed with request
        response = await call_next(request)
        return response

def create_ip_middleware(allowed_ips=None, allowed_networks=None):
    """
    Factory function to create IP restriction middleware.

    Args:
        allowed_ips: List of allowed IP addresses
        allowed_networks: List of allowed IP networks

    Returns:
        IPRestrictionMiddleware instance
    """
    return IPRestrictionMiddleware(
        allowed_ips=allowed_ips,
        allowed_networks=allowed_networks
    )
