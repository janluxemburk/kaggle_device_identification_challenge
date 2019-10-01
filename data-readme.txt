Every record in the dataset is a network scan of a device in JSON format (one scan per line) with the fields:

*) mac: MAC address of the device
*) ip: local IP
*) device_id: unique id of the scan
*) device_class: this is what you need to predict
*) services: the list of open ports discovered by our network scanner
    - first, the scanner tries to discover open ports
        - What is a port: https://en.wikipedia.org/wiki/Port_(computer_networking)
        - How it can be done technically: https://nmap.org/book/man-port-scanning-techniques.html

*) upnp: UPnP response
    - then the scanner tries to find and interrogate the UPnP server (not every device has UPnP server running)
    - if you know nothing about UPnP, this is a good place to start:
        http://www.upnp-hacks.org/upnp.html

*) mdns: mDNS response
    - the scanner tries to get the list of registered services with mDNS
        - https://angus.nyc/2013/zero-conf-bootstrapping-the-network-layer/

*) ssdp: SSDP broadcasts
*) dhcp: DHCP broadcasts
    - When scanning a device the scanner is sniffing the network for DHCP and SSDP broadcasts sent by the device
        - https://en.wikipedia.org/wiki/Dynamic_Host_Configuration_Protocol
        - https://williamboles.me/discovering-whats-out-there-with-ssdp/
        - http://lets-start-to-learn.blogspot.com/2015/02/dhcp-fingerprinting.html


Possible device classes with examples:

*) AUDIO: audio devices (smart speakers, internet radio)
*) GAME_CONSOLE: Xbox, Sony PlayStation
*) HOME_AUTOMATION: smart home IoT devices (lighting, switches, alarms)
*) IP_PHONE: office IP phones and conference systems
*) MEDIA_BOX: set-top boxes and streaming dongles
*) MOBILE: smartphones and tablets
*) NAS: network storage devices
*) PC: desktop computers and laptops
*) PRINTER: printers, scanners
*) SURVEILLANCE: IP/security cameras
*) TV
*) VOICE_ASSISTANT: Amazon Echo, Google Home
*) GENERIC_IOT: other IoT devices (medical and industrial equipment, smart scales, sleeping monitors)

