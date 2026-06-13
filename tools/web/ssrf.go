package web

import (
	"fmt"
	"net"
	"net/url"
	"strings"
)

// privateNetworks defines CIDR ranges considered internal/private.
var privateNetworks = []string{
	"127.0.0.0/8",    // loopback
	"10.0.0.0/8",     // RFC 1918
	"172.16.0.0/12",  // RFC 1918
	"192.168.0.0/16", // RFC 1918
	"169.254.0.0/16", // link-local
	"0.0.0.0/8",      // "this" network / unspecified; routes to loopback on Linux
	"100.64.0.0/10",  // CGNAT (RFC 6598)
	"192.0.0.0/24",   // IETF protocol assignments (RFC 6890)
	"::1/128",        // IPv6 loopback
	"fc00::/7",       // IPv6 unique local
	"fe80::/10",      // IPv6 link-local
	"::/128",         // IPv6 unspecified
}

var parsedPrivateNets []*net.IPNet

func init() {
	for _, cidr := range privateNetworks {
		_, ipNet, err := net.ParseCIDR(cidr)
		if err == nil {
			parsedPrivateNets = append(parsedPrivateNets, ipNet)
		}
	}
}

// isPrivateIP returns true if the IP address is in a private/internal range.
func isPrivateIP(ip net.IP) bool {
	for _, n := range parsedPrivateNets {
		if n.Contains(ip) {
			return true
		}
	}
	return false
}

// ValidateURLNotInternal checks that a URL does not point to an internal/private
// network address. This prevents SSRF attacks that could access cloud metadata
// endpoints, localhost services, or internal network resources.
func ValidateURLNotInternal(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}

	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("URL has no host")
	}

	// Block common metadata hostnames
	lower := strings.ToLower(host)
	if lower == "metadata.google.internal" || lower == "metadata" {
		return fmt.Errorf("access to internal metadata endpoint is blocked")
	}

	// Resolve hostname to IP addresses
	ips, err := net.LookupIP(host)
	if err != nil {
		return fmt.Errorf("cannot resolve hostname %q — blocking to prevent SSRF", host)
	}

	for _, ip := range ips {
		if isPrivateIP(ip) {
			return fmt.Errorf("access to internal address %s (%s) is blocked", host, ip)
		}
	}

	return nil
}
