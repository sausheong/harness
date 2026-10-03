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
	"198.18.0.0/15",  // benchmarking (RFC 2544)
	"224.0.0.0/4",    // IPv4 multicast
	"240.0.0.0/4",    // reserved, incl. 255.255.255.255 broadcast
	"ff00::/8",       // IPv6 multicast
	"2001::/32",      // Teredo (embeds an obfuscated IPv4)
	"64:ff9b:1::/48", // local-use NAT64 (RFC 8215)
}

// embeddedV4Nets are IPv6 prefixes that carry an IPv4 address translators may
// route to: the embedded address is re-checked so e.g. 64:ff9b::7f00:1 (NAT64
// for 127.0.0.1) is blocked while a public embedded address stays allowed.
var embeddedV4Nets = []struct {
	net    *net.IPNet
	offset int // byte offset of the embedded IPv4 address
}{
	{mustCIDR("64:ff9b::/96"), 12}, // NAT64 well-known prefix
	{mustCIDR("2002::/16"), 2},     // 6to4
	{mustCIDR("::/96"), 12},        // IPv4-compatible (deprecated)
}

func mustCIDR(s string) *net.IPNet {
	_, n, err := net.ParseCIDR(s)
	if err != nil {
		panic(err)
	}
	return n
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
	if ip.To4() == nil {
		if ip16 := ip.To16(); ip16 != nil {
			for _, e := range embeddedV4Nets {
				if e.net.Contains(ip16) {
					return isPrivateIP(net.IP(ip16[e.offset : e.offset+4]))
				}
			}
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
