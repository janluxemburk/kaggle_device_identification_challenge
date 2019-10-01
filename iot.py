# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:03:25 2019

"""


import re
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from itertools import chain
from sklearn.model_selection import train_test_split

device_classes = [
    "AUDIO", "GAME_CONSOLE", "HOME_AUTOMATION", "IP_PHONE", "MEDIA_BOX",
    "MOBILE", "NAS", "PC", "PRINTER", "SURVEILLANCE", "TV", "VOICE_ASSISTANT", "GENERIC_IOT"
]

mac_8 = {}
mac_10 = {}
mac_13 = {}

with open("macaddress.io-db.json", encoding="utf8") as f:
    for line in f:
        manufacturer = json.loads(line)
        prefix = manufacturer["oui"].lower()
        prefix_len = len(prefix)
        if prefix_len == 8:
            mac_8[prefix] = manufacturer["companyName"],  manufacturer["countryCode"]
        elif prefix_len == 10:
            mac_10[prefix] =  manufacturer["companyName"], manufacturer["countryCode"]
        elif prefix_len == 13:
            mac_13[prefix] =  manufacturer["companyName"], manufacturer["countryCode"]


def get_manufacturer_and_country(row):
    mac = row["mac"].lower()
    if (mac == "00:00:00:00:00:00"):
        return np.nan, np.nan
    prefix8 = mac[:8]
    prefix10 = mac[:10]
    prefix13 = mac[:13]

    if prefix8 in mac_8:
        return mac_8[prefix8]
    elif prefix10 in mac_10:
        return mac_10[prefix10]
    elif prefix13 in mac_13:
        return mac_13[prefix13]
    return np.nan, np.nan


known_ports = ['10000tcp', '10001tcp', '10080tcp', '10243tcp', '1024tcp', '1025tcp', '1026tcp', '1027tcp', '1028tcp', '1029tcp', '1030tcp', '1031tcp', '1032tcp', '1033tcp', '1034tcp', '1035tcp', '1036tcp', '1037tcp', '1038tcp', '1039tcp', '1040tcp', '1040udp', '1041tcp', '1042tcp', '1043tcp', '1044tcp', '1045tcp', '1046tcp', '1047tcp', '1048tcp', '1049tcp', '1050tcp', '1051tcp', '1052tcp', '1053tcp', '1054tcp', '10554tcp', '1055tcp', '1056tcp', '1060tcp', '1061tcp', '1062tcp', '1063tcp', '1065tcp', '1066tcp', '1068tcp', '1069tcp', '1071tcp', '1072tcp', '1073tcp', '1074tcp', '1075tcp', '1076tcp', '1078tcp', '1079tcp', '1080tcp', '1081tcp', '1082tcp', '1083tcp', '1084tcp', '1086tcp', '1087tcp', '1089tcp', '1090tcp', '1092tcp', '1094tcp', '1096tcp', '1098tcp', '1100tcp', '1101tcp', '1102tcp', '1103tcp', '1106tcp', '1107tcp', '1108tcp', '1109tcp', '110tcp', '1110tcp', '1111tcp', '1112tcp', '1113tcp', '1115tcp', '1116tcp', '1118tcp', '1119tcp', '111tcp', '111udp', '1120tcp', '1122tcp', '1123tcp', '1124tcp', '1125tcp', '1126tcp', '1127tcp', '1129tcp', '1130tcp', '1134tcp', '1136tcp', '1139tcp', '1142tcp', '1143tcp', '1144tcp', '1145tcp', '1146tcp', '1149tcp', '1151tcp', '1156tcp', '1158tcp', '1159tcp', '1160tcp', '1164tcp', '1166tcp', '1167tcp', '1169tcp', '1170tcp', '1171tcp', '1173tcp', '1174tcp', '1175tcp', '1178tcp', '1179tcp', '1180tcp', '1181tcp', '1183tcp', '1184tcp', '1185tcp', '1186tcp', '1188tcp', '1189tcp', '1190tcp', '1195tcp', '1196tcp', '1197tcp', '1199tcp', '1200tcp', '1201tcp', '1202tcp', '1204tcp', '1209tcp', '1211tcp', '1212tcp', '1214tcp', '1216tcp', '1217tcp', '1218tcp', '1219tcp', '1220tcp', '1221tcp', '1222tcp', '1225tcp', '1226tcp', '1227tcp', '1228tcp', '1229tcp', '1230tcp', '1231tcp', '1233tcp', '1234tcp', '1235tcp', '1237tcp', '1238tcp', '1239tcp', '123udp', '1240tcp', '1242tcp', '1243tcp', '1244tcp', '1245tcp', '1246tcp', '1247tcp', '1248tcp', '1249tcp', '1250tcp', '1251tcp', '1252tcp', '1253tcp', '1254tcp', '1260tcp', '1261tcp', '1263tcp', '1264tcp', '1267tcp', '1268tcp', '1269tcp', '1272tcp', '1273tcp', '1274tcp', '1275tcp', '1277tcp', '1278tcp', '1280tcp', '1281tcp', '1282tcp', '12837tcp', '1284tcp', '1285tcp', '1286tcp', '1287tcp', '1288tcp', '1289tcp', '1290tcp', '1291tcp', '1293tcp', '1296tcp', '1297tcp', '13000tcp', '1300tcp', '1301tcp', '1303tcp', '1304tcp', '1305tcp', '1306tcp', '1307tcp', '1308tcp', '1309tcp', '1310tcp', '1312tcp', '1314tcp', '1315tcp', '1317tcp', '1318tcp', '1319tcp', '1321tcp', '1323tcp', '1324tcp', '1327tcp', '1329tcp', '1332tcp', '1333tcp', '1334tcp', '1335tcp', '1336tcp', '1338tcp', '1339tcp', '1340tcp', '1341tcp', '1342tcp', '1343tcp', '1344tcp', '1345tcp', '1347tcp', '1348tcp', '1350tcp', '1351tcp', '1353tcp', '1354tcp', '1355tcp', '1356tcp', '1357tcp', '1358tcp', '1359tcp', '135tcp', '135udp', '1360tcp', '1361tcp', '1364tcp', '1365tcp', '1367tcp', '1368tcp', '1369tcp', '1370tcp', '1373tcp', '1375tcp', '1376tcp', '137udp', '1380tcp', '1381tcp', '1384tcp', '1386tcp', '1387tcp', '1388tcp', '1389tcp', '138udp', '1390tcp', '1392tcp', '1394tcp', '1397tcp', '1398tcp', '1399tcp', '139tcp', '1400tcp', '1401tcp', '1402tcp', '1403tcp', '1404tcp', '1406tcp', '1407tcp', '1409tcp', '1410tcp', '1411tcp', '1412tcp', '1413tcp', '1414tcp', '1416tcp', '1417tcp', '1418tcp', '1419tcp', '1420tcp', '1422tcp', '1423tcp', '1424tcp', '1425tcp', '1426tcp', '1428tcp', '1429tcp', '1430tcp', '1431tcp', '1432tcp', '1433tcp', '1434tcp', '1434udp', '1435tcp', '1437tcp', '1438tcp', '1440tcp', '1441tcp', '1442tcp', '1443tcp', '1445tcp', '1446tcp', '1447tcp', '1449tcp', '1450tcp', '1451tcp', '1452tcp', '1453tcp', '1456tcp', '1458tcp', '1462tcp', '1465tcp', '1466tcp', '1467tcp', '1468tcp', '1469tcp', '1471tcp', '1472tcp', '1473tcp', '1475tcp', '1476tcp', '1477tcp', '1478tcp', '1482tcp', '1483tcp', '1485tcp', '1486tcp', '1488tcp', '1490tcp', '1492tcp', '1493tcp', '1494tcp', '1498tcp', '1499tcp', '1500tcp', '1501tcp', '1502tcp', '1503tcp', '1505tcp', '1507tcp', '1508tcp', '1509tcp', '1512tcp', '1513tcp', '1514tcp', '1516tcp', '1519tcp', '1520tcp', '1522tcp', '1523tcp', '1524tcp', '1525tcp', '1527tcp', '1528tcp', '1530tcp', '1532tcp', '1533tcp', '1534tcp', '1535tcp', '1536tcp', '1537tcp', '1538tcp', '1539tcp', '1541tcp', '1542tcp', '1545tcp', '1546tcp', '1548tcp', '1549tcp', '1553tcp', '1554tcp', '1555tcp', '1556tcp', '1557tcp', '1559tcp', '1560tcp', '1561tcp', '1563tcp', '1564tcp', '1565tcp', '1568tcp', '1569tcp', '1570tcp', '1572tcp', '1573tcp', '1574tcp', '1575tcp', '1576tcp', '1577tcp', '1579tcp', '1580tcp', '1583tcp', '1584tcp', '1585tcp', '1586tcp', '1587tcp', '1589tcp', '1590tcp', '1591tcp', '1592tcp', '1593tcp', '1594tcp', '1596tcp', '1597tcp', '1599tcp', '1600tcp', '16021tcp', '1602tcp', '1603tcp', '1604tcp', '1605tcp', '1607tcp', '1608tcp', '1609tcp', '1610tcp', '1611tcp', '1612tcp', '1613tcp', '1615tcp', '1616tcp', '1617tcp', '161tcp', '161udp', '1620tcp', '1621tcp', '1623tcp', '1624tcp', '1625tcp', '1626tcp', '1629tcp', '162udp', '1631tcp', '1632tcp', '1633tcp', '1634tcp', '1638tcp', '1640tcp', '1641tcp', '1642tcp', '1643tcp', '1644tcp', '1646tcp', '1647tcp', '1648tcp', '1650tcp', '1651tcp', '1652tcp', '1653tcp', '1654tcp', '1655tcp', '1656tcp', '1657tcp', '1660tcp', '1662tcp', '1663tcp', '1664tcp', '1669tcp', '1671tcp', '1672tcp', '1674tcp', '1676tcp', '1677tcp', '1679tcp', '1683tcp', '1685tcp', '16881tcp', '1688tcp', '1689tcp', '1690tcp', '1692tcp', '1694tcp', '1695tcp', '1696tcp', '1697tcp', '1698tcp', '16992tcp', '16993tcp', '1699tcp', '1700tcp', '1701tcp', '1702tcp', '1703tcp', '1704tcp', '1705tcp', '1708tcp', '1709tcp', '1712tcp', '1714tcp', '1715tcp', '1716tcp', '1717tcp', '1718tcp', '1719tcp', '1720tcp', '1721tcp', '1722tcp', '1723tcp', '1728tcp', '1729tcp', '1734tcp', '1738tcp', '1739tcp', '1742tcp', '1743tcp', '1744tcp', '1745tcp', '1746tcp', '1748tcp', '1749tcp', '1750tcp', '1751tcp', '1752tcp', '1753tcp', '1754tcp', '1756tcp', '1757tcp', '1758tcp', '1759tcp', '1761tcp', '1762tcp', '1763tcp', '1765tcp', '1766tcp', '1767tcp', '1769tcp', '1770tcp', '1771tcp', '1772tcp', '1775tcp', '1776tcp', '1778tcp', '1779tcp', '1780tcp', '1782tcp', '1783tcp', '1784tcp', '1785tcp', '1786tcp', '1787tcp', '1788tcp', '1791tcp', '1792tcp', '1794tcp', '1795tcp', '1796tcp', '1799tcp', '1800tcp', '1801tcp', '1802tcp', '1804tcp', '1805tcp', '1806tcp', '1807tcp', '1808tcp', '1809tcp', '1813tcp', '1814tcp', '1815tcp', '1817tcp', '1819tcp', '1820tcp', '1821tcp', '1822tcp', '1824tcp', '1825tcp', '1826tcp', '1827tcp', '1829tcp', '1830tcp', '1831tcp', '1833tcp', '1836tcp', '1837tcp', '1838tcp', '1839tcp', '1840tcp', '1842tcp', '1843tcp', '1844tcp', '1846tcp', '1848tcp', '1849tcp', '1850tcp', '1854tcp', '1855tcp', '1857tcp', '1858tcp', '1859tcp', '1860tcp', '1861tcp', '1862tcp', '1863tcp', '1864tcp', '1866tcp', '1867tcp', '1868tcp', '1870tcp', '1872tcp', '1874tcp', '1875tcp', '1877tcp', '1878tcp', '1880tcp', '1881tcp', '1883tcp', '1884tcp', '1885tcp', '1886tcp', '1887tcp', '1888tcp', '1889tcp', '1891tcp', '1892tcp', '1894tcp', '1896tcp', '1897tcp', '1898tcp', '1899tcp', '1900tcp', '1900udp', '1901tcp', '1902tcp', '1905tcp', '1906tcp', '1908tcp', '1909tcp', '1910tcp', '1911tcp', '1912tcp', '1915tcp', '1916tcp', '1917tcp', '1919tcp', '1920tcp', '1921tcp', '1922tcp', '1923tcp', '1924tcp', '1925tcp', '1926tcp', '1927tcp', '1931tcp', '1933tcp', '1934tcp', '1936tcp', '1938tcp', '1939tcp', '1940tcp', '1941tcp', '1944tcp', '1945tcp', '1946tcp', '1947tcp', '1948tcp', '1949tcp', '1953tcp', '1955tcp', '1956tcp', '1957tcp', '1959tcp', '1960tcp', '1963tcp', '1965tcp', '1969tcp', '1971tcp', '1972tcp', '1973tcp', '1974tcp', '1976tcp', '1977tcp', '1978tcp', '1980tcp', '1981tcp', '1984tcp', '1985tcp', '1987tcp', '1988tcp', '1989tcp', '1990tcp', '1990udp', '1991tcp', '1992tcp', '1995tcp', '1996tcp', '1997tcp', '19999tcp', '1999tcp', '20005tcp', '2000tcp', '2001tcp', '2002tcp', '2003udp', '2004tcp', '2005tcp', '2006tcp', '2007tcp', '2010tcp', '2011tcp', '20121tcp', '2012tcp', '2013tcp', '2014tcp', '2015tcp', '2018tcp', '2020tcp', '2022tcp', '2023tcp', '2024tcp', '2026tcp', '2027tcp', '2028tcp', '2032tcp', '2034tcp', '2037tcp', '2038tcp', '2039tcp', '2040tcp', '2041tcp', '2042tcp', '2044tcp', '2046tcp', '2047tcp', '2049tcp', '2121tcp', '21tcp', '22tcp', '2323tcp', '2343tcp', '23tcp', '24322udp', '25454udp', '25tcp', '2869tcp', '2870tcp', '2875tcp', '30000tcp', '3000tcp', '3031tcp', '3128tcp', '32400tcp', '32469tcp', '32498tcp', '3262tcp', '32764tcp', '3283udp', '33344tcp', '3389tcp', '3493tcp', '3500tcp', '3540udp', '3587tcp', '3600tcp', '3689tcp', '3697tcp', '3702udp', '37215tcp', '37904tcp', '38388tcp', '38400tcp', '38520tcp', '389tcp', '3986tcp', '4000tcp', '4070tcp', '4071tcp', '41800tcp', '42300tcp', '427tcp', '427udp', '4301tcp', '4301udp', '4433tcp', '443tcp', '4443tcp', '4446udp', '4456tcp', '445tcp', '445udp', '4500udp', '46263tcp', '47365tcp', '4747tcp', '47984tcp', '47989tcp', '49152tcp', '49153tcp', '49154tcp', '49155tcp', '49156tcp', '49157tcp', '49160tcp', '49161tcp', '49163tcp', '49164tcp', '49165tcp', '49166tcp', '49170tcp', '49172tcp', '49174tcp', '49175tcp', '49176tcp', '49197tcp', '49664tcp', '49665tcp', '49666tcp', '49667tcp', '49668tcp', '49669tcp', '50000tcp', '50001tcp', '50003tcp', '5000tcp', '5001tcp', '5002tcp', '5004udp', '5005tcp', '5005udp', '5006tcp', '500udp', '50201tcp', '5040tcp', '5050udp', '5060tcp', '514tcp', '515tcp', '5200tcp', '52235tcp', '52323tcp', '52396tcp', '53413udp', '5350udp', '5351udp', '5353udp', '5355udp', '5357tcp', '53tcp', '53udp', '54243tcp', '5431udp', '54380tcp', '546udp', '548tcp', '548udp', '54921tcp', '55000tcp', '55001tcp', '554tcp', '554udp', '55555tcp', '5555tcp', '56071tcp', '56789tcp', '56790tcp', '5683udp', '5684udp', '5800tcp', '58080tcp', '5900tcp', '59777tcp', '60000tcp', '60006tcp', '6000tcp', '6011tcp', '6012tcp', '6013tcp', '6014tcp', '625tcp', '6281tcp', '631tcp', '64321tcp', '6466tcp', '6517tcp', '6600tcp', '6666tcp', '6690tcp', '67udp', '68udp', '69udp', '7000tcp', '7547tcp', '7676tcp', '7677tcp', '7678tcp', '7680tcp', '7777tcp', '8000tcp', '8001tcp', '8008tcp', '8009tcp', '8010tcp', '8060tcp', '8080tcp', '8081tcp', '8083tcp', '8085tcp', '8089tcp', '8090tcp', '8091tcp', '8099tcp', '80tcp', '8100tcp', '8181tcp', '81tcp', '8200tcp', '8266tcp', '8289tcp', '8291tcp', '82tcp', '83tcp', '8443tcp', '8554tcp', '85tcp', '8610tcp', '8611tcp', '8612tcp', '8722tcp', '873tcp', '8766tcp', '8880tcp', '8888tcp', '8895tcp', '88tcp', '8913tcp', '8965tcp', '8987tcp', '8989tcp', '9000tcp', '9001tcp', '9002tcp', '9080tcp', '9081tcp', '9090tcp', '9095tcp', '9096tcp', '9097tcp', '9098tcp', '9100tcp', '9119tcp', '9197tcp', '9295tcp', '9400tcp', '9500tcp', '9777udp', '9876tcp', '9900tcp', '993tcp', '9955tcp', '9955udp', '9956udp', '9998tcp', '9999tcp', '9999udp', '9tcp', '9udp']
known_dhcp_params= ['1', '2', '3', '4', '5', '6', '7', '9', '11', '12', '13', '15', '16', '17', '18', '22', '23', '26', '28', '29', '31', '33', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '54', '58', '59', '60', '66', '67', '69', '70', '72', '78', '79', '81', '88', '95', '97', '100', '101', '119', '120', '121', '125', '128', '129', '130', '131', '132', '133', '134', '135', '142', '143', '144', '150', '151', '153', '154', '159', '160', '161', '190', '204', '249', '252']
known_dhcp_classids = ['BLACKBERRY', 'IP_PHONE', 'LINUX', 'PRINTER', 'UDHCP_AUDIO', 'UDHCP_TV', 'android-dhcp-6.0', 'android-dhcp-6.0.1', 'android-dhcp-7.0', 'android-dhcp-7.1.2', 'android-dhcp-8.0.0', 'android-dhcp-8.1.0', 'android-dhcp-9', 'dhcpcd-5.2.10:linux-3.8.13+:armv7l:marvell berlin soc (flattened device tree)', 'dhcpcd-5.2.10:linux-3.8.13+:armv7l:mv88de3108', 'dhcpcd-5.2.10:linux-4.1.16-mrvl:aarch64', 'dhcpcd-5.2.10:linux-4.9.93:aarch64', 'dhcpcd-5.5.6', 'dhcpcd-6.10.1:linux-3.8.13-mrvl:armv7l:marvell', 'emlab', 'ip-stb', 'ms-uc-client', 'msft 5.0', 'stb_bytel', 'udhcp 0.9.9-pre', 'udhcp 1.12.1', 'udhcp 1.20.2', 'udhcp 1.22.1', 'udhcp 1.23.2', 'udhcp 1.24.1']
known_mdsn = ['_CGI._tcp.local.', '_JB2XiOSApp._tcp.local.', '_WECB_AEI._tcp.local.', '_acer._tcp.local.', '_acp-sync._tcp.local.', '_adb._tcp.local.', '_adisk._tcp.local.', '_afpovertcp._tcp.local.', '_airdrop._tcp.local.', '_airplay._tcp.local.', '_airport._tcp.local.', '_alljoyn._tcp.local.', '_alljoyn._udp.local.', '_amzn-wplay._tcp.local.', '_androidtvremote._tcp.local.', '_anymote._tcp.local.', '_apple-lgremote._tcp.local.', '_apple-midi._udp.local.', '_apple-mobdev._tcp.local.', '_apple-mobdev2._tcp.local.', '_appletv-v2._tcp.local.', '_aptoide-rmtinst._tcp.local.', '_arduino._tcp.local.', '_atc._tcp.local.', '_axis-video._tcp.local.', '_blackarmor1dconfig._tcp.local.', '_blackarmor1dinfo._udp.local.', '_blackarmor2dconfig._tcp.local.', '_blackarmor2dinfo._udp.local.', '_blackarmor4dconfig._tcp.local.', '_blackarmor4dinfo._udp.local.', '_bose-passport._tcp.local.', '_bp2p._tcp.local.', '_canon-bjnp1._tcp.local.', '_canon-chmp._tcp.local.', '_coap._udp.local.', '_companion-link._tcp.local.', '_cros_p2p._tcp.local.', '_csco-sb._tcp.local.', '_daap._tcp.local.', '_dacp._tcp.local.', '_dcamera._tcp.local.', '_dcp._tcp.local.', '_device-info._tcp.local.', '_dhnap._tcp.local.', '_display._tcp.local.', '_edcp._udp.local.', '_eppc._tcp.local.', '_fax-ipp._tcp.local.', '_fbx-api._tcp.local.', '_fbx-devel._tcp.local.', '_ftp._tcp.local.', '_gamecenter._tcp.local.', '_gasetup._udp.local.', '_googlecast._tcp.local.', '_googlezone._tcp.local.', '_hap._tcp.local.', '_harman_002._tcp.local.', '_hearing._tcp.local.', '_hid._udp.local.', '_home-sharing._tcp.local.', '_homekit._tcp.local.', '_http-alt._tcp.local.', '_http._tcp.local.', '_https._tcp.local.', '_icc-service._tcp.local.', '_ipp-tls._tcp.local.', '_ipp._tcp.local.', '_ipps._tcp.local.', '_iri._tcp.local.', '_lap._tcp.local.', '_leap._tcp.local.', '_lutron._tcp.local.', '_makerbot-jsonrpc._tcp.local.', '_mdns._udp.local.', '_mediaremotetv._tcp.local.', '_mfi-config._tcp.local.', '_mpd._tcp.local.', '_mqtt._tcp.local.', '_mycanal._tcp.local.', '_nanoleafapi._tcp.local.', '_nanoleafms._tcp.local.', '_net-assistant._udp.local.', '_nfs._tcp.local.', '_ni-logos._tcp.local.', '_nipca._tcp.local.', '_nut._tcp.local.', '_nuvomaster._tcp.local.', '_nuvoplayer._tcp.local.', '_nv_shield_remote._tcp.local.', '_nvstream._tcp.local.', '_nvstream_dbd._tcp.local.', '_odisk._tcp.local.', '_odproxy._tcp.local.', '_osc._udp.local.', '_parentcontrol._tcp.local.', '_pblipc._tcp.local.', '_pdl-datastream._tcp.local.', '_philipstv_rpc._tcp.local.', '_philipstv_s_rpc._tcp.local.', '_plexmediasvr._tcp.local.', '_presence._tcp.local.', '_print-caps._tcp.local.', '_printer._tcp.local.', '_privet._tcp.local.', '_psia._tcp.local.', '_qboxcluster._tcp.local.', '_qdiscover._tcp.local.', '_qmobile._tcp.local.', '_raop._tcp.local.', '_rdlink._tcp.local.', '_readynas._tcp.local.', '_rfb._tcp.local.', '_ros-master._tcp.local.', '_rp-hsd._tcp.local.', '_rp-media._tcp.local.', '_rsp._tcp.local.', '_rsync._tcp.local.', '_rtsp._tcp.local.', '_runestone._udp.local.', '_satellite_info._tcp.local.', '_scan-target._tcp.local.', '_scanner._tcp.local.', '_servermgr._tcp.local.', '_sftp-ssh._tcp.local.', '_sftp._tcp.local.', '_silhouettelink._tcp.local.', '_sironaxray._tcp.local.', '_sleep-proxy._udp.local.', '_smb._tcp.local.', '_soundtouch._tcp.local.', '_spotify-connect._tcp.local.', '_ssh._tcp.local.', '_sub._apple-mobdev2._tcp.local.', '_sueGrouping._tcp.local.', '_sueS800Device._tcp.local.', '_tdrsservice._tcp.local.', '_tdrsservices._tcp.local.', '_teamviewer._tcp.local.', '_telnet._tcp.local.', '_tftp._udp.local.', '_tivo-device._tcp.local.', '_tivo-mindrpc._tcp.local.', '_tivo-remote._tcp.local.', '_tivo-videos._tcp.local.', '_tivo-videostream._tcp.local.', '_tivo-xcode._tcp.local.', '_touch-able._tcp.local.', '_tunnel._tcp.local.', '_tw-multipeer._tcp.local.', '_udisks-ssh._tcp.local.', '_upnp._tcp.local.', '_uscan._tcp.local.', '_uscans._tcp.local.', '_vhusb._tcp.local.', '_video._tcp.local.', '_viziocast._tcp.local.', '_wd-2go._tcp.local.', '_webdav._tcp.local.', '_webdavs._tcp.local.', '_workstation._tcp.local.', '_xbmc-events._udp.local.', '_xbmc-jsonrpc-h._tcp.local.', '_xbmc-jsonrpc._tcp.local.', '_xboxcol._tcp.local.', '_xrdovertcp._tcp.local.', '_yv-bridge._tcp.local.', '_zound._tcp.local.']
known_companies = ['ASUSTek Computer Inc', 'Abus Security-Center GmbH & Co KG', 'Advidia', 'Aio Lcd PC BU / Tpv', 'Amazon Tech Inc', 'Amazon.com, Llc', 'Apple, Inc', 'Arcadyan Corp', 'Arcadyan Tech Corp', 'Arris Group, Inc', 'Asiarock Tech Ltd', 'Askey Computer Corp', 'Axis Communications AB', 'AzureWave Tech Inc', 'Belkin International Inc', 'Bematech International Corp', "Biostar Microtech Int'l Corp", 'Bixolon Co, Ltd', 'BlackBerry Rts', 'Brother industries, Ltd', 'Chicony Electronics Co, Ltd', 'Compal Information (Kunshan) Co, Ltd', 'D&M Holdings Inc', 'D-Link Corp', 'D-Link International', 'Davicom Semiconductor, Inc', 'Digibras Industria DO Brasils/A', 'Edimax Tech Co Ltd', 'Elitegroup Computer Systems Co, Ltd', 'Espressif Inc', 'Freebox Sas', 'Freecom Tech GmbH', 'Frontier Silicon Ltd', 'Fuji-Xerox Co Ltd', 'Fujitsu Tech Solutions GmbH', 'Gemtek Tech Co, Ltd', 'Giga-Byte Tech Co, Ltd', 'Google, Inc', 'Guangdong Oppo Mobile Telecommunications Corp, Ltd', 'Hangzhou Hikvision Digital Tech Co, Ltd', 'Hewlett Packard', 'Hewlett Packard Enterprise', 'Hon Hai Precision Ind. Co, Ltd', 'Hui Zhou Gaoshengda Tech Co, Ltd', 'Icp Electronics Inc', 'Ieee Registration Authority', 'Inspur (Shandong) Electronic Information Co, Ltd', 'Konica Minolta Holdings, Inc', 'Kreatel Communications AB', 'Kyocera Display Corp', 'LA Cie Group S.A.', 'LG Electronics', 'LG Electronics Inc', 'LG innotek', 'LG-Ericsson Co, Ltd', 'LT Security Inc', 'Lexmark International, Inc', 'Lifi Labs Management Pty Ltd', 'LiteON', 'Liteon Tech Corp', 'Loxone Electronics GmbH', 'Lsd Science and Tech Co, Ltd', 'MakerBot Industries', "Micro-Star Int'L Co, Ltd", 'Micro-Star International Co, Ltd', 'Microsoft', 'Microsoft Corp', 'Mitel Corp', 'Mitsumi Electric Co, Ltd', 'Murata Manufacturing Co, Ltd', 'Netatmo', 'Netgear', 'Nintendo Co, Ltd', 'Nokia Corp', 'NuVo Tech Llc', 'Nvidia', 'Oki Electric Industry Co, Ltd', 'OnePlus Tech (Shenzhen) Co, Ltd', 'OnePlus Tech (Shenzhen) Ltd', 'Onkyo Corp', 'Palladium Energy Eletronica DA Amazonia Ltda', 'Panasonic Appliances Co', 'Panasonic Communications Co, Ltd', 'Panasonic Corp Avc Networks Co', 'Philips Electronics Nederland BV', 'Philips Lighting BV', 'Phorus', 'Pioneer Corp', 'Private', 'Qingdao Hisense Communications Co, Ltd', 'Realtek Semiconductor Corp', 'Resideo', 'Ricoh Co Ltd', 'Roku, Inc', 'Sagemcom Broadband Sas', 'Samjin Co, Ltd', 'Samsung Electro Mechanics Co, Ltd', 'Samsung Electro-Mechanics(Thailand)', 'Samsung Electronics', 'Samsung Electronics Co, Ltd', 'Segate Tech Llc', 'Sharp Corp', 'Shenzhen Bilian Electronic Co，Ltd', 'Sirona Dental Systems GmbH & Co KG', 'Smd Informatica S.A.', 'Sonos, Inc', 'Sony Corp', 'Sony Interactive Entertainment Inc', 'Sony Visual Products Inc', 'Swann communications Pty Ltd', 'Synology Inc', 'TP Vision Belgium N.V. - innovation site Brugge', 'TP Vision Belgium NV', 'TP-Link Tech Co, Ltd', 'TRENDnet, Inc', 'Taiyo Yuden Co, Ltd', 'Thecus Tech Corp', 'Tokyo Electric Co, Ltd', 'Universal Global Scientific Industrial Co, Ltd', 'Wistron Corp', 'Wistron Neweb Corp', 'Withings', 'Xerox Corp', 'Xiamen Yealink Network Tech Co, Ltd', 'Xiaomi Communications Co Ltd', 'Yealink(Xiamen) Network Tech Co, Ltd', 'Zavio Inc', 'Zyxel Communications Corp', 'ecobee Inc']
known_countries = ['CN', 'US', 'TW', 'CA', 'FR', 'KR', 'JP', 'NL', 'AU', 'FI', 'TH', 'BE', 'DE', 'SE', 'AT', 'GB', 'SG', 'PT', 'VG', 'BR']
known_ssdp = ['__agent__AVAST', '__agent__AVG Secure Browser/71.0.693.100 Windows', '__agent__AVG Secure Browser/72.0.718.83 Windows', '__agent__CHROMEOS', '__agent__Chromium/71.0.3578.98 Windows', '__agent__CocCoc/71.0.3578.126 Windows', '__agent__Eset UPnP/1.1', '__agent__GOOGLE-CHROME', '__agent__LGE_DL', '__agent__LINUX', '__agent__MACOS', '__agent__SONOS', '__agent__UDAP/2.0', '__agent__UPnP/1.0 DLNADOC/1.50 Kodi', '__agent__UPnP/1.0 DLNADOC/1.50 Platinum/1.0.4.2', '__agent__UPnP/1.0 DLNADOC/1.50 Platinum/1.0.5.13', '__agent__WINDOWS', '__agent__dLeyna/0.5.0 GSSDP/0.14.12', '__agent__master-process GSSDP/1.0.1', '__agent__unix/1.0 UPnP/1.1 TVHeadend/0.0.0~unknown', '__nt__NAS', '__nt__f7ea4045-41c0-4866-8def-a9d023c05378', '__nt__fbx:devel', '__nt__hnap:networkcamera', '__nt__http://www.konicaminolta.com/service/openapi-3-0', '__nt__nanoleaf_aurora:light', '__nt__raumfeld:timeserver', '__nt__roku:ecp', '__nt__upnp:acn-com:device:utc', '__nt__upnp:acn-com:device:utclcdi', '__nt__upnp:acn-com:device:utctrigger', '__nt__upnp:acn-com:service:utcinhome', '__nt__upnp:acn-com:service:utclcdiapi', '__nt__upnp:acn-com:service:utcrcemulation', '__nt__upnp:rootdevice', '__nt__urn:alphanetworks:service:basicservice', '__nt__urn:axis-com:service:basicservice', '__nt__urn:belkin:device:controllee', '__nt__urn:belkin:device:dimmer', '__nt__urn:belkin:device:insight', '__nt__urn:belkin:device:lightswitch', '__nt__urn:belkin:device:netcamsensor', '__nt__urn:belkin:service:basicevent', '__nt__urn:belkin:service:crockpotevent', '__nt__urn:belkin:service:deviceevent', '__nt__urn:belkin:service:deviceinfo', '__nt__urn:belkin:service:firmwareupdate', '__nt__urn:belkin:service:insight', '__nt__urn:belkin:service:jardenevent', '__nt__urn:belkin:service:manufacture', '__nt__urn:belkin:service:metainfo', '__nt__urn:belkin:service:remoteaccess', '__nt__urn:belkin:service:rules', '__nt__urn:belkin:service:smartsetup', '__nt__urn:belkin:service:timesync', '__nt__urn:belkin:service:wifisetup', '__nt__urn:bouygues-telecom-com:device:bboxtv', '__nt__urn:bouygues-telecom-com:service:probeprovider', '__nt__urn:cellvision:service:null', '__nt__urn:dial-multiscreen-org:device:dial', '__nt__urn:dial-multiscreen-org:device:dialreceiver', '__nt__urn:dial-multiscreen-org:service:dial', '__nt__urn:dmc-samsung-com:device:syncserver', '__nt__urn:dmc-samsung-com:service:syncmanager', '__nt__urn:lge-com:device:multiroomspk', '__nt__urn:lge-com:device:sstdevice', '__nt__urn:lge-com:service:webos-second-screen', '__nt__urn:lge-com:service:x_lg_wfdisplay', '__nt__urn:lge:device:tv', '__nt__urn:lge:service:virtualsvc', '__nt__urn:microsoft-com:device:hsbsserver', '__nt__urn:microsoft-com:service:discoveryservice', '__nt__urn:microsoft.com:service:x_ms_mediareceiverregistrar', '__nt__urn:nw-dtv-jp:service:inettv_appliance_service', '__nt__urn:orange.com:service:remotecontrol', '__nt__urn:panasonic-com:device:p00proavcontroller', '__nt__urn:panasonic-com:device:p00remotecontroller', '__nt__urn:panasonic-com:service:p00dimoraremotecontrol', '__nt__urn:panasonic-com:service:p00networkcontrol', '__nt__urn:panasonic-com:service:p00panasonic_session_service', '__nt__urn:panasonic-com:service:p00proavcontrolservice', '__nt__urn:panasonic-com:service:p01netcamservice', '__nt__urn:pv-com:device:nmcqueuehandlerdevice', '__nt__urn:pv-com:device:proxyserver', '__nt__urn:rockchip-com:service:remotecontrol', '__nt__urn:samsung.com:device:maintvserver2', '__nt__urn:samsung.com:device:remotecontrolreceiver', '__nt__urn:samsung.com:device:screensharing', '__nt__urn:samsung.com:service:maintvagent2', '__nt__urn:samsung.com:service:multiscreenservice', '__nt__urn:samsung.com:service:screensharingservice', '__nt__urn:samsung.com:service:testrcrservice', '__nt__urn:sbox-dev-com:device:sbox', '__nt__urn:sbox-dev-com:service:sboxcontrolservice', '__nt__urn:schemas-awox-com:service:x_servicemanager', '__nt__urn:schemas-cipa-jp:device:dpsprinter', '__nt__urn:schemas-cipa-jp:service:dpsconnectionmanager', '__nt__urn:schemas-cyberlink-com:service:remotecontrol', '__nt__urn:schemas-denon-com:device:act-denon', '__nt__urn:schemas-denon-com:device:aiosdevice', '__nt__urn:schemas-denon-com:device:aiosservices', '__nt__urn:schemas-denon-com:service:act', '__nt__urn:schemas-denon-com:service:errorhandler', '__nt__urn:schemas-denon-com:service:groupcontrol', '__nt__urn:schemas-denon-com:service:zonecontrol', '__nt__urn:schemas-dm-holdings-com:service:x_htmlpagehandler', '__nt__urn:schemas-dm-holdings-com:service:x_wholehomeaudio', '__nt__urn:schemas-frontier-silicon-com:fs_reference:fsapi', '__nt__urn:schemas-frontier-silicon-com:hama_001:fsapi', '__nt__urn:schemas-frontier-silicon-com:invalid:fsapi', '__nt__urn:schemas-frontier-silicon-com:medion_001:fsapi', '__nt__urn:schemas-frontier-silicon-com:sangean_001:fsapi', '__nt__urn:schemas-frontier-silicon-com:targa_001:fsapi', '__nt__urn:schemas-frontier-silicon-com:technisat_001:fsapi', '__nt__urn:schemas-frontier-silicon-com:undok:fsapi', '__nt__urn:schemas-frontier-silicon-com:undok:fsaudsync', '__nt__urn:schemas-microsoft-com:service:null', '__nt__urn:schemas-nuvotechnologies-com:device:zone', '__nt__urn:schemas-orange-com:service:x_orangestbremotecontrol', '__nt__urn:schemas-raumfeld-com:device:configdevice', '__nt__urn:schemas-raumfeld-com:device:raumfelddevice', '__nt__urn:schemas-raumfeld-com:service:configservice', '__nt__urn:schemas-raumfeld-com:service:raumfeldgenerator', '__nt__urn:schemas-raumfeld-com:service:setupservice', '__nt__urn:schemas-rvualliance-org:service:streamsplicing', '__nt__urn:schemas-smsc-com:service:x_wholehomeaudio', '__nt__urn:schemas-sonos-com:service:queue', '__nt__urn:schemas-sony-com:service:group', '__nt__urn:schemas-sony-com:service:ircc', '__nt__urn:schemas-sony-com:service:multichannel', '__nt__urn:schemas-sony-com:service:scalarwebapi', '__nt__urn:schemas-sony-com:service:x_telepathy', '__nt__urn:schemas-tencent-com:service:qplay', '__nt__urn:schemas-upnp-org:basic', '__nt__urn:schemas-upnp-org:device:airborne', '__nt__urn:schemas-upnp-org:device:basic', '__nt__urn:schemas-upnp-org:device:basic:1.0', '__nt__urn:schemas-upnp-org:device:binarylight', '__nt__urn:schemas-upnp-org:device:dcs-3220', '__nt__urn:schemas-upnp-org:device:dial', '__nt__urn:schemas-upnp-org:device:digitalsecuritycamera', '__nt__urn:schemas-upnp-org:device:embeddednetdevice', '__nt__urn:schemas-upnp-org:device:hvac_system', '__nt__urn:schemas-upnp-org:device:internetgatewaydevice', '__nt__urn:schemas-upnp-org:device:manageabledevice', '__nt__urn:schemas-upnp-org:device:mediarenderer', '__nt__urn:schemas-upnp-org:device:mediaserver', '__nt__urn:schemas-upnp-org:device:nas', '__nt__urn:schemas-upnp-org:device:nastorage', '__nt__urn:schemas-upnp-org:device:networkstoragedevice', '__nt__urn:schemas-upnp-org:device:printer', '__nt__urn:schemas-upnp-org:device:remoteuiserver', '__nt__urn:schemas-upnp-org:device:renderingcontrol', '__nt__urn:schemas-upnp-org:device:tvdevice', '__nt__urn:schemas-upnp-org:device:wanconnectiondevice', '__nt__urn:schemas-upnp-org:device:wandevice', '__nt__urn:schemas-upnp-org:device:wdnas', '__nt__urn:schemas-upnp-org:device:winas', '__nt__urn:schemas-upnp-org:device:wirelessnetworkcamera', '__nt__urn:schemas-upnp-org:device:zoneplayer', '__nt__urn:schemas-upnp-org:service:abcontrol', '__nt__urn:schemas-upnp-org:service:alarmclock', '__nt__urn:schemas-upnp-org:service:audioin', '__nt__urn:schemas-upnp-org:service:avtransport', '__nt__urn:schemas-upnp-org:service:basicmanagement', '__nt__urn:schemas-upnp-org:service:changeip', '__nt__urn:schemas-upnp-org:service:configurationmanagement', '__nt__urn:schemas-upnp-org:service:connectionmanager', '__nt__urn:schemas-upnp-org:service:contentdirectory', '__nt__urn:schemas-upnp-org:service:deviceproperties', '__nt__urn:schemas-upnp-org:service:deviceprotection', '__nt__urn:schemas-upnp-org:service:digitalsecuritycamerasettings', '__nt__urn:schemas-upnp-org:service:embeddednetdevicecontrol', '__nt__urn:schemas-upnp-org:service:energymanagement', '__nt__urn:schemas-upnp-org:service:groupmanagement', '__nt__urn:schemas-upnp-org:service:grouprenderingcontrol', '__nt__urn:schemas-upnp-org:service:hisenselancontrol', '__nt__urn:schemas-upnp-org:service:htcontrol', '__nt__urn:schemas-upnp-org:service:ipccontrol', '__nt__urn:schemas-upnp-org:service:layer3forwarding', '__nt__urn:schemas-upnp-org:service:musicservices', '__nt__urn:schemas-upnp-org:service:nascontrol', '__nt__urn:schemas-upnp-org:service:nastorage', '__nt__urn:schemas-upnp-org:service:p00remoteaccess', '__nt__urn:schemas-upnp-org:service:printbasic', '__nt__urn:schemas-upnp-org:service:printenhanced', '__nt__urn:schemas-upnp-org:service:remoteuiserver', '__nt__urn:schemas-upnp-org:service:renderingcontrol', '__nt__urn:schemas-upnp-org:service:systemproperties', '__nt__urn:schemas-upnp-org:service:virtuallinein', '__nt__urn:schemas-upnp-org:service:virtualremotecontrol', '__nt__urn:schemas-upnp-org:service:wancommoninterfaceconfig', '__nt__urn:schemas-upnp-org:service:wanipconnection', '__nt__urn:schemas-upnp-org:service:wiautoconfig', '__nt__urn:schemas-upnp-org:service:zonegrouptopology', '__nt__urn:schemas-wd-com:device:wdnas-dev_0064', '__nt__urn:schemas-wd-com:device:wdnas-dev_0094', '__nt__urn:schemas-wd-com:device:wdnas-dev_0096', '__nt__urn:schemas-wifialliance-org:device:wfadevice', '__nt__urn:schemas-wifialliance-org:service:wfawlanconfig', '__nt__urn:smarthomealliance-org:device:gateway', '__nt__urn:smarthomealliance-org:service:smarthomeservice', '__nt__urn:smartspeaker-audio:service:speakergroup', '__nt__urn:www-seagate-com:device:banas', '__nt__urn:www-seagate-com:device:nasos', '__nt__urn:www-seagate-com:device:scnas', '__nt__urn:www-seagate-com:service:nas', '__nt__uuid:4ca69fea-391e-a2c3-0000-000038b27980', '__nt__uuid:4d696e69-444c-164e-9d41-00089bd875c9', '__nt__uuid:4d696e69-444c-164e-9d41-00184dffff07', '__nt__uuid:4e50646a-b607-4ecb-9676-8dc10abe8a5f', '__nt__uuid:55076f6e-6b79-1d65-a4eb-00089bd875c9', '__nt__uuid:5d076f6e-6b79-1d65-a4eb-00089bd875c9', '__nt__uuid:c1fd12b2-d954-4dba-9e92-a697e1558fb4', '__nt__uuid:upnp-ds-2cv2q21fd-iw-1_0-214701551', '__server__1.1 DLNADOC/1.50 UPnP/1.0 SITEVIEW/1.0', '__server__3.4.6-generic Microsoft-Windows/6.1 Windows-Media-Player-DMS/12.0.7601.17514 DLNADOC/1.50 UPnP/1.0 QNAPDLNA/1.0', '__server__AFICIO', '__server__ARRIS DIAL/1.7.2 UPnP/1.0 ARRIS Settop Box', '__server__ARRIS DIAL/2.1 UPnP/1.0 ARRIS Settop Box', '__server__Allegro-Software-RomPager/5.41 UPnP/1.0 ARRIS Settop Box', '__server__Android/1.6 UPnP/1.0 Huey Sample DMR/0.1', '__server__Bose-Software/5.40 UPnP/1.1 SoundTouch Music Server/2.0', '__server__Canon IJ-UPnP/1.0 UPnP/1.0 UPnP-Device-Host/1.0', '__server__Cellvision UPnP/1.0', '__server__DENON', '__server__DLNADOC/1.50 Linux/3.10.0 UPnP/1.0 RKDLNALib/2.0', '__server__Embedded UPnP/1.0', '__server__Embedded/1.0 UPnP/1.0 IPCamera-UPnP/1.0', '__server__Embedded/1.0 UPnP/1.0 IPCamera/1.1', '__server__Embedded/1.0 UPnP/1.0 Network Camera/1.0', '__server__FOS/1.0 UPnP/1.0 Jupiter/6.5', '__server__FOS/1.0 UPnP/1.0 NS/1.0', '__server__FedoraCore/2 UPnP/1.0 MINT-X/1.8.1', '__server__FreeBSD/8.0 UPnP/1.0 Panasonic-MIL-DLNA-SV/1.0', '__server__IPBRIDGE', '__server__IPI/1.0 UPnP/1.0 DLNADOC/1.50', '__server__KDL', '__server__KnOS/3.2 UPnP/1.0 DMP/3.5', '__server__LG-MS/9741 Linux/2.6.35 UPnP/1.0 DLNADOC/1.50 LGE_DLNA_SDK/1.5.0', '__server__LINUX UPnP/1.0 Avega AIOS/', '__server__LOXONE', '__server__LPUX', '__server__Linux 2.5 SHP/1.1 Gateway/1.0', '__server__Linux/ UPnP/1.0 DLNADOC/1.50 LogitechMediaServer/7.7.6/1452060463', '__server__Linux/2.6 UPnP/1.0 HT-ZF9/0.01', '__server__Linux/2.6 UPnP/1.0 HT-ZF9/1.0', '__server__Linux/2.6 UPnP/1.0 fbxupnpav/1.0', '__server__Linux/2.6.22 UPnP/1.1 JetHeadInc/1', '__server__Linux/2.6.30.9 UPnP/1.0', '__server__Linux/2.6.33-rc4 UPnP/1.0 MediabolicUPnP/1.8.225', '__server__Linux/2.6.33N7700, UPnP/1.0, Intel SDK for UPnP devices /1.2', '__server__Linux/2.6.35 UPnP/1.0 DLNADOC/1.50 INTEL_NMPR/2.0 LGE_DLNA_SDK/1.5.0', '__server__Linux/2.6.35.11-83.fc14.i686 UPnP/1.0 miniupnpd/1.0', '__server__Linux/2.6.35.6-45.fc14.i686 UPnP/1.0 miniupnpd/1.0', '__server__Linux/2.6.37 UPnP/1.0 GUPnP/0.20.8', '__server__Linux/2.6.39.4.ps-110224-lg1152 UPnP/1.0 DLNADOC/1.50 INTEL_NMPR/2.0 LGE_DLNA_SDK/1.5.0', '__server__Linux/2.x.x, UPnP/1.0, pvConnect UPnP SDK/1.0', '__server__Linux/2.x.x, UPnP/1.0, pvConnect UPnP SDK/1.0, Twonky UPnP SDK/1.1', '__server__Linux/2.x.x, UPnP/1.0, pvConnect UPnP SDK/1.0, TwonkyMedia UPnP SDK/1.1', '__server__Linux/3.0.31 UPnP/1.0 Cling/2.0', '__server__Linux/3.10 UPnP/1.0 AV Receiver/2.0', '__server__Linux/3.10 UPnP/1.0 Sony-AVR/2.0', '__server__Linux/3.10 UPnP/1.0 Sony-BDP/2.0', '__server__Linux/3.10 UPnP/1.0 Sony-BDV/2.0', '__server__Linux/3.10 UPnP/1.0 Sony-HTS/2.0', '__server__Linux/3.10.19-22 UPnP/1.0 LG Smart TV/1.0', '__server__Linux/3.10.46 UPnP/1.0 Cling/2.0', '__server__Linux/3.10.54 UPnP/1.0 Cling/2.0', '__server__Linux/3.10.54 UPnP/1.0 Teleal-Cling/1.0', '__server__Linux/3.10.61 UPnP/1.0 Cling/2.0', '__server__Linux/3.10.79 UPnP/1.0 Cling/2.0', '__server__Linux/3.10.79 UPnP/1.0 Teleal-Cling/1.0', '__server__Linux/3.14.29 UPnP/1.0 Cling/2.0', '__server__Linux/3.14.29 UPnP/1.0 CyberLinkJava/1.8', '__server__Linux/3.14.43+ UPnP/1.0 GUPnP/0.20.18', '__server__Linux/3.18.71+ UPnP/1.0 GUPnP/1.0.2', '__server__Linux/3.19.0 UPnP/1.0 GUPnP/0.20.13', '__server__Linux/3.4.0-perf-ge9ecc17bf39 UPnP/1.0 Cling/2.0', '__server__Linux/4.0 UPnP/1.0 Panasonic-MIL-DLNA-SV/1.0', '__server__Linux/4.0.9 Raumfeld/1.0 TimeService/1.0', '__server__Linux/4.0.9 UPnP/1.0 GUPnP/1.0.1', '__server__Linux/4.0.9 UPnP/1.0 GUPnP/1.0.3', '__server__Linux/4.4.120 UPnP/1.0 Cling/2.0', '__server__Linux/4.9.109-tegra-g2dafed3 UPnP/1.0 Teleal-Cling/1.0', '__server__Linux/9.0 UPnP/1.0 PROTOTYPE/1.0', '__server__Linux/i686 UPnP/1,0 DLNADOC/1.50 LGE WebOS TV/Version 0.9', '__server__Linux/i686 UPnP/1.0 DLNADOC/1.50 Platinum/1.0.3.0', '__server__Linux2.6/0.0 UPnP/1.0 PhilipsIntelSDK/1.4 ', '__server__Linux2.6/0.0 UPnP/1.0 PhilipsIntelSDK/1.4 DLNADOC/1.50', '__server__MICROSTACK', '__server__MPC', '__server__Microsoft-Windows-NT/5.1 UPnP/1.0 UPnP-Device-Host/1.0', '__server__Microsoft-Windows/10.0 UPnP/1.0 UPnP-Device-Host/1.0', '__server__Microsoft-Windows/6.2 UPnP/1.0 UPnP-Device-Host/1.0', '__server__Microsoft-Windows/6.3 UPnP/1.0 UPnP-Device-Host/1.0', '__server__NETWORK-PRINTER-SERVER', '__server__NFLC/2.3 UPnP/1.0 DLNADOC/1.50', '__server__NFLC/3.0 UPnP/1.0 DLNADOC/1.50', '__server__Original', '__server__PORTABLE_SDK', '__server__POSIX, UPnP/1.0', '__server__POSIX, UPnP/1.0 UPnP Stack/6.37.14.87', '__server__PRINTER', '__server__Platform 1.0 His/1.0 UPnP/1.0 DLNADOC/1.50', '__server__RAIDiator OS UPnP/1.0 upnpd/1.0', '__server__READYDLNA', '__server__ReadyNASOS UPnP/1.0', '__server__SHP, UPnP/1.0, Samsung UPnP SDK/1.0', '__server__SOFTATHOME', '__server__SONOS', '__server__SRS', '__server__SYNOLOGY', '__server__TORRENT', '__server__TP-LINK', '__server__Telepathy/1 UPnP/1.0 Telepathy/1', '__server__UPnP/1.0 AwoX/1.1', '__server__UPnP/1.0 DLNADOC/1.50 AirReceiver/1.0.3.0', '__server__UPnP/1.0 DLNADOC/1.50 Platinum/1.0.4.11', '__server__UPnP/1.0 DLNADOC/1.50 Platinum/1.0.5.13', '__server__UPnP/1.0 UPnP/1.0 UPnP-Device-Host/1.0', '__server__Unspecified, UPnP/1.0, Unspecified', '__server__WINDOWS', '__server__WebOS/1.5 UPnP/1.0', '__server__WebOS/4.0.0 UPnP/1.0', '__server__WebOS/4.0.3 UPnP/1.0', '__server__WebOS/4.1.0 UPnP/1.0', '__server__Windows2000/0.0 UPnP/1.0 PhilipsIntelSDK/1.4 DLNADOC/1.50', '__server__XboxUpnp/0.1 UPnP/1.0 Xbox/2.0.14042.0', '__server__windows/6.1 IntelUSBoverIP:1/1', '__st__media:router', '__st__roku:ecp', '__st__ssdp:all', '__st__udap:rootservice', '__st__upnp:rootdevice', '__st__urn:belkin:device:**', '__st__urn:belkin:device:controllee', '__st__urn:belkin:device:insight', '__st__urn:belkin:device:lightswitch', '__st__urn:belkin:device:sensor', '__st__urn:belkin:service:basicevent', '__st__urn:dial-multiscreen-org:device:dial', '__st__urn:dial-multiscreen-org:service:dial', '__st__urn:dslforum-org:device:internetgatewaydevice', '__st__urn:lge-com:service:webos-second-screen', '__st__urn:mdx-netflix-com:service:target', '__st__urn:microsoft windows peer name resolution protocol: v4:ipv6:linklocal', '__st__urn:microsoft.com:service:x_ms_mediareceiverregistrar', '__st__urn:pv-com:device:nmcqueuehandlerdevice', '__st__urn:pv-com:device:proxyserver', '__st__urn:rvualliance-org:device:rvuserver', '__st__urn:samsung.com:device:remotecontrolreceiver', '__st__urn:samsung.com:device:screensharing', '__st__urn:schemas-ce-org:device:remoteuiserverdevice', '__st__urn:schemas-denon-com:device:act-denon', '__st__urn:schemas-digion-com:service:x_accesscontrol', '__st__urn:schemas-digion-com:service:x_landisk', '__st__urn:schemas-dm-holdings-com:service:x_wholehomeaudio', '__st__urn:schemas-frontier-silicon-com:undok:fsaudsync', '__st__urn:schemas-heosbydenon-com:device:wirelessrangeextenderdevice', '__st__urn:schemas-intel-com:service:powermanagement', '__st__urn:schemas-microsoft-com:service:pbda:tuner', '__st__urn:schemas-opencable-com:service:tuner', '__st__urn:schemas-raumfeld-com:device:configdevice', '__st__urn:schemas-upnp-org:device:basic', '__st__urn:schemas-upnp-org:device:internetgatewaydevice', '__st__urn:schemas-upnp-org:device:manageabledevice', '__st__urn:schemas-upnp-org:device:mediarenderer', '__st__urn:schemas-upnp-org:device:mediaserver', '__st__urn:schemas-upnp-org:device:remoteuiclientdevice', '__st__urn:schemas-upnp-org:device:remoteuiserver', '__st__urn:schemas-upnp-org:device:remoteuiserverdevice', '__st__urn:schemas-upnp-org:device:wanconnectiondevice', '__st__urn:schemas-upnp-org:device:zoneplayer', '__st__urn:schemas-upnp-org:service:avtransport', '__st__urn:schemas-upnp-org:service:contentdirectory', '__st__urn:schemas-upnp-org:service:wanipconnection', '__st__urn:schemas-upnp-org:service:wanpppconnection', '__st__urn:schemas-wifialliance-org:device:wfadevice', '__st__urn:ses-com:device:satipserver', '__st__urn:smartspeaker-audio:service:speakergroup', '__st__uuid:12345678-0000-0000-0000-00000000abcd', '__st__uuid:75802409-bccb-40e7-8e6c-fa095ecce13e']
known_upnp = ['__TV__', '__description__1-bay personal cloud storage (gen2)', '__description__1.3m', '__description__2-bay personal cloud storage', '__description__2.4ghz wireless internet camera', '__description__CAMERA', '__description__READY-NAS-OS', '__description__SONOS', '__description__[cube hd ipcam professional]', '__description__airreceiver - media renderer', '__description__airreceiver - youtube dial server', '__description__ais virtual remote control.', '__description__allplay capable network audio module.', '__description__av receiver', '__description__av surround receiver', '__description__axis a1001 network door controller', '__description__axis a8004-ve network video door station', '__description__axis a8105-e network video door station', '__description__belkin insight 1.0', '__description__belkin plugin dimmer 1.0', '__description__belkin plugin socket 1.0', '__description__belkin wemo wi-fi to zigbee bridge', '__description__blackarmor nas 2d', '__description__blackarmor nas 4d', '__description__bose soundtouch rest music server', '__description__bose soundtouch wireless streaming audio device', '__description__bouyguestelecom hmb4213h', '__description__bravia', '__description__bridgeco digital media adapter with upnp', '__description__c3000-100nas', '__description__c3700-100nas', '__description__camera', '__description__cyberlink powerdvd dlna renderer', '__description__cyberlink upnp media server', '__description__cyberlink upnp media server ultra', '__description__d-link hd ir outdoor network camera', '__description__device management', '__description__dial server', '__description__digital media client', '__description__digital media player', '__description__digital media renderer', '__description__digital video recorder', '__description__dlna 1.5 compliant media renderer from softathome', '__description__dlna 1.5 digital media server', '__description__eedomus home automation box', "__description__es file explorer's dlna render on android", '__description__eureka dongle - chromecast', '__description__freebox upnp renderer', '__description__freecom musicpal', '__description__full hd wireless n cube network camera', '__description__hd cube network camera', '__description__hisense mediarenderer 1.0', '__description__home audio system', '__description__intel av media renderer device', '__description__internet camera', '__description__internet camera.', '__description__ip camera', '__description__ipi media renderer', '__description__laciemediaserver', '__description__lg music flow', '__description__lg netcast 4.0 dmrplus', '__description__lg sst device', '__description__lg webostv dmrplus', '__description__logitech media server upnp/dlna plugin', '__description__media server', '__description__mediarenderer 1.0', '__description__medion life e85052', '__description__medion life p85024', '__description__medion life p85035', '__description__microchip digital media adapter with upnp', '__description__my book live network storage', '__description__naim mu-so all-in-one audio player', '__description__nas device', '__description__nas326', '__description__network audio player', '__description__network camera', '__description__network cd receiver', '__description__network receiver', '__description__nmc queue handler', '__description__panasonic network camera', '__description__personal audio system', '__description__personal cloud', '__description__philips hue personal wireless lighting', '__description__philips tv server', '__description__phorus-renderer', '__description__pkm2cn1828br', '__description__plex media server', '__description__poe internet camera', '__description__qnapdlna on turbonas', '__description__radiostation', '__description__readydlna', '__description__readydlna on readynas raidiator os', '__description__rockchip media renderer, dlna(dmr)', '__description__roku streaming player network media', '__description__samsung bd ns', '__description__samsung bd rcr', '__description__samsung dtv maintvserver2', '__description__samsung dtv rcr', '__description__samsung printer device ', '__description__samsung rvu uhd tv 2015', '__description__samsung rvu uhd tv 2016', '__description__samsung rvu uhd tv 2017', '__description__samsung soundbar dmr', '__description__samsung tv dmr', '__description__samsung tv ns', '__description__samsung tv rcr', '__description__samsung tv screensharing', '__description__seagate blackarmor nas 2d', '__description__seagate blackarmor nas 4d', '__description__seagate central nas 1d', '__description__seagate central shared storage dlna', '__description__seagatemediaserver', '__description__sensor camera', '__description__serviio, a dlna media server', '__description__shares user defined folders and files to other universal plug and play media devices.', '__description__smart box', '__description__sonos beam', '__description__sonos beam media renderer', '__description__sonos beam media server', '__description__sonos boost', '__description__sonos bridge', '__description__sonos connect', '__description__sonos connect media renderer', '__description__sonos connect media server', '__description__sonos connect:amp', '__description__sonos connect:amp media renderer', '__description__sonos connect:amp media server', '__description__sonos one', '__description__sonos one media renderer', '__description__sonos one media server', '__description__sonos play:1', '__description__sonos play:1 media renderer', '__description__sonos play:1 media server', '__description__sonos play:3', '__description__sonos play:3 media renderer', '__description__sonos play:3 media server', '__description__sonos play:5', '__description__sonos play:5 media renderer', '__description__sonos play:5 media server', '__description__sonos playbar', '__description__sonos playbar media renderer', '__description__sonos playbar media server', '__description__sonos sub', '__description__sonos sub media renderer', '__description__sonos zp100', '__description__sonos zp100 media renderer', '__description__sonos zp100 media server', '__description__sony 2015 2k tv', '__description__sony 2015 4k rvu tv', '__description__sony 2015 4k tv', '__description__sony 2015 rvu tv', '__description__sony 2017 4k rvu tv', '__description__sony 2017 4k tv', '__description__swisscom tv 2.0 dlna renderer', '__description__synology dlna/upnp media server', '__description__synology nas', '__description__twonkymedia server', '__description__twonkyserver (linux, t2)', '__description__twonkyserver (linux-pc, t2)', '__description__upnp media renderer 1.0', '__description__upnp mg2900 series dps printer 1 ', '__description__upnp mg3600 series dps printer 1 ', '__description__upnp mg5700 series dps printer 1 ', '__description__upnp mx920 series dps printer 1 ', '__description__virtual media player', '__description__wd my cloud', '__description__wifi/dab+/fm radio', '__description__windows media player renderer', '__description__wireless internet camera', '__description__wireless pan/tilt internet camera', '__description__xbox 360', '__devicetype__upnp:acn-com:device:utc:2', '__devicetype__upnp:acn-com:device:utclcdi:1', '__devicetype__upnp:acn-com:device:utctrigger:1', '__devicetype__urn:belkin:device:bridge:1', '__devicetype__urn:belkin:device:controllee:1', '__devicetype__urn:belkin:device:dimmer:1', '__devicetype__urn:belkin:device:insight:1', '__devicetype__urn:belkin:device:lightswitch:1', '__devicetype__urn:belkin:device:netcamsensor:1', '__devicetype__urn:belkin:device:sensor:1', '__devicetype__urn:bouygues-telecom-com:device:bboxtv:1', '__devicetype__urn:dial-multiscreen-org:device:dial:1', '__devicetype__urn:dial-multiscreen-org:device:dialreceiver:1', '__devicetype__urn:dial-multiscreen-org:service:dial:1', '__devicetype__urn:dmc-samsung-com:device:syncserver:1', '__devicetype__urn:lge-com:device:multiroomspk:1', '__devicetype__urn:lge-com:device:sstdevice:1', '__devicetype__urn:lge:device:tv:1', '__devicetype__urn:panasonic-com:device:p00proavcontroller:1', '__devicetype__urn:panasonic-com:device:p00remotecontroller:1', '__devicetype__urn:pv-com:device:nmcqueuehandlerdevice:1', '__devicetype__urn:roku-com:device:player:1-0', '__devicetype__urn:samsung.com:device:maintvserver2:1', '__devicetype__urn:samsung.com:device:remotecontrolreceiver:1', '__devicetype__urn:samsung.com:device:screensharing:1', '__devicetype__urn:sbox-dev-com:device:sbox:1', '__devicetype__urn:schemas-bmlinks-jp:device:bmlinks:1', '__devicetype__urn:schemas-cipa-jp:device:dpsprinter:1', '__devicetype__urn:schemas-cyberlink-com:device:sparkdevice:1', '__devicetype__urn:schemas-denon-com:device:act-denon:1', '__devicetype__urn:schemas-denon-com:device:aiosdevice:1', '__devicetype__urn:schemas-denon-com:device:aiosservices:1', '__devicetype__urn:schemas-nuvotechnologies-com:device:zone:1', '__devicetype__urn:schemas-raumfeld-com:device:configdevice:1', '__devicetype__urn:schemas-raumfeld-com:device:raumfelddevice:1', '__devicetype__urn:schemas-upnp-org:device:basic:1', '__devicetype__urn:schemas-upnp-org:device:basic:1.0', '__devicetype__urn:schemas-upnp-org:device:binarylight:1', '__devicetype__urn:schemas-upnp-org:device:dial:1', '__devicetype__urn:schemas-upnp-org:device:digitalsecuritycamera:1', '__devicetype__urn:schemas-upnp-org:device:embeddednetdevice:1', '__devicetype__urn:schemas-upnp-org:device:hvac_system:1', '__devicetype__urn:schemas-upnp-org:device:internetgatewaydevice:1', '__devicetype__urn:schemas-upnp-org:device:manageabledevice:2', '__devicetype__urn:schemas-upnp-org:device:mediarenderer:1', '__devicetype__urn:schemas-upnp-org:device:mediarenderer:2', '__devicetype__urn:schemas-upnp-org:device:mediarenderer:3', '__devicetype__urn:schemas-upnp-org:device:mediaserver:1', '__devicetype__urn:schemas-upnp-org:device:mediaserver:3', '__devicetype__urn:schemas-upnp-org:device:nas:1', '__devicetype__urn:schemas-upnp-org:device:nastorage:1', '__devicetype__urn:schemas-upnp-org:device:networkstoragedevice:1', '__devicetype__urn:schemas-upnp-org:device:printer:1', '__devicetype__urn:schemas-upnp-org:device:tvdevice:1', '__devicetype__urn:schemas-upnp-org:device:wanconnectiondevice:1', '__devicetype__urn:schemas-upnp-org:device:wandevice:1', '__devicetype__urn:schemas-upnp-org:device:wdnas:1', '__devicetype__urn:schemas-upnp-org:device:winas:1', '__devicetype__urn:schemas-upnp-org:device:wirelessnetworkcamera:1', '__devicetype__urn:schemas-upnp-org:device:zoneplayer:1', '__devicetype__urn:schemas-wd-com:device:wdnas-dev_0064:1', '__devicetype__urn:schemas-wd-com:device:wdnas-dev_0094:1', '__devicetype__urn:schemas-wd-com:device:wdnas-dev_0096:1', '__devicetype__urn:schemas-wifialliance-org:device:wfadevice:1', '__devicetype__urn:upnp-logitech-com:device:securitydevice:1', '__devicetype__urn:www-seagate-com:device:banas:2', '__devicetype__urn:www-seagate-com:device:nasos:1', '__devicetype__urn:www-seagate-com:device:scnas:1', '__manufacturer__LG-ELECTRONICS', '__manufacturer__abus security-center', '__manufacturer__access co., ltd.', '__manufacturer__amazon', '__manufacturer__amlogic corporation', '__manufacturer__arcadyan', '__manufacturer__arris', '__manufacturer__auna', '__manufacturer__axis', '__manufacturer__belkin international inc.', '__manufacturer__bose', '__manufacturer__bose corporation', '__manufacturer__brickcom corporation', '__manufacturer__bridgeco ag, switzerland', '__manufacturer__canon inc.', '__manufacturer__connected object', '__manufacturer__cyberlink corporation', '__manufacturer__d-link', '__manufacturer__d-link corporation', '__manufacturer__dell electronics', '__manufacturer__denon', '__manufacturer__dual', '__manufacturer__es', '__manufacturer__freebox sas', '__manufacturer__freecom', '__manufacturer__frontier silicon ltd', '__manufacturer__fujitsu technology solutions gmbh', '__manufacturer__google inc.', '__manufacturer__hama', '__manufacturer__hikvision', '__manufacturer__hp', '__manufacturer__inspur', '__manufacturer__lacie', '__manufacturer__lanier', '__manufacturer__lautsprecher teufel gmbh', '__manufacturer__logitech', '__manufacturer__loxone electronics gmbh', '__manufacturer__lynx technology', '__manufacturer__marantz', '__manufacturer__medion ag', '__manufacturer__microchip', '__manufacturer__microsoft', '__manufacturer__microsoft corporation', '__manufacturer__naim audio ltd.', '__manufacturer__netgear', '__manufacturer__nuvo', '__manufacturer__nvidia', '__manufacturer__oem', '__manufacturer__onkyo', '__manufacturer__onkyo & pioneer corporation', '__manufacturer__onkyo and pioneer', '__manufacturer__packetvideo', '__manufacturer__panasonic', '__manufacturer__petr nejedly', '__manufacturer__philips', '__manufacturer__phorus', '__manufacturer__pioneer corporation', '__manufacturer__plex, inc.', '__manufacturer__qnap systems, inc.', '__manufacturer__qualcomm allplay', '__manufacturer__ricoh', '__manufacturer__roberts radio limited', '__manufacturer__rockchip', '__manufacturer__roku', '__manufacturer__royal philips electronics', '__manufacturer__sagem', '__manufacturer__samsung', '__manufacturer__samsung electronics', '__manufacturer__samsung-electronics', '__manufacturer__sangean radio limited', '__manufacturer__savin', '__manufacturer__sbox dev', '__manufacturer__seagate', '__manufacturer__seagate corporation', '__manufacturer__seagate technology plc', '__manufacturer__softathome', '__manufacturer__softmedia inc.', '__manufacturer__sonos, inc.', '__manufacturer__sony', '__manufacturer__sony corporation', '__manufacturer__swisscom', '__manufacturer__synology', '__manufacturer__synology inc', '__manufacturer__technisat', '__manufacturer__thecus', '__manufacturer__tp-link', '__manufacturer__upnp', '__manufacturer__utc', '__manufacturer__western digital corporation', '__manufacturer__www.thecus.com', '__manufacturer__xerox corporation', '__manufacturer__xerox electronics', '__manufacturer__zavio', '__manufacturer__zyxel', '__modelname__ samsung m408x series ', '__modelname__ seagate personal cloud', '__modelname__*AVR', '__modelname__*m-cr611', '__modelname__*nr1506', '__modelname__*nr1606', '__modelname__*rcd-n9', '__modelname__-', '__modelname__2-way audio surveillance camera', '__modelname__2.4ghz wireless internet camera', '__modelname__2016allxx', '__modelname__32lk615bpsb', '__modelname__32s301', '__modelname__32s305', '__modelname__40s305', '__modelname__43lk5700psc', '__modelname__43lk5750psa', '__modelname__43lk5900pla', '__modelname__43uk6200pla', '__modelname__43uk6300mlb', '__modelname__43uk6400plf', '__modelname__49s405', '__modelname__49uk6400plf', '__modelname__50uk6520psa', '__modelname__55r6+', '__modelname__55s401', '__modelname__55s405', '__modelname__55uk6200pla', '__modelname__55uk6950plb', '__modelname__65s401', '__modelname__AFICIO', '__modelname__CAMERA', '__modelname__DCS', '__modelname__DENON-AVR', '__modelname__DS', '__modelname__DS-', '__modelname__HIKVISION', '__modelname__KDL', '__modelname__NAS', '__modelname__SEAGATE', '__modelname__SOUND', '__modelname__TS', '__modelname__TX-NR', '__modelname__XBR', '__modelname__afta', '__modelname__aftb', '__modelname__aftm', '__modelname__aftmm', '__modelname__aftn', '__modelname__afts', '__modelname__aftt', '__modelname__airreceiver', '__modelname__av renderer', '__modelname__avr-1912', '__modelname__axis a1001', '__modelname__axis a8004-ve', '__modelname__axis a8105-e', '__modelname__bdp-s3700', '__modelname__blackarmor nas 2d', '__modelname__blackarmor nas 4d', '__modelname__blu-ray disc player', '__modelname__bose home speaker 500', '__modelname__bose soundbar 500', '__modelname__bose soundbar 700', '__modelname__bose soundtouch rest music server', '__modelname__bouygteltv', '__modelname__bravia 2015', '__modelname__bravia 4k 2015', '__modelname__bravia 4k gb', '__modelname__bravia 4k gb atv3', '__modelname__bridge', '__modelname__c3000-100nas', '__modelname__c3700-100nas', '__modelname__cb-101ap', '__modelname__chromecast ultra', '__modelname__cyberlink media server', '__modelname__cyberlink powerdvd', '__modelname__d6210 2mp dome camera', '__modelname__day/night surveillance camera', '__modelname__denon ceol', '__modelname__digitalmediaadapterupnp', '__modelname__dimmer', '__modelname__dir3100', '__modelname__dra-n5', '__modelname__eedomus', '__modelname__es file explorer', '__modelname__eureka dongle', '__modelname__freebox', '__modelname__freecom musicpal', '__modelname__google home hub', '__modelname__h.264 1080p wired ir-bullet camera, poe', '__modelname__h.264 2-megapixel d/n bullet camera w/ audio, i/o(2/2), poe', '__modelname__h.264 720p wired d/n indoor dome, i/o, poe', '__modelname__h.264 720p wired ir cube camera , poe', '__modelname__heos 1', '__modelname__heos homecinema', '__modelname__ht-ct790', '__modelname__ht-ct800', '__modelname__ht-nt5', '__modelname__ht-st5000', '__modelname__ht-xt2', '__modelname__ht-xt3', '__modelname__ht-zf9', '__modelname__hw-k950', '__modelname__hw-ms650', '__modelname__icu100', '__modelname__ihd92', '__modelname__insight', '__modelname__insignia ns-32dr310na17', '__modelname__ip815', '__modelname__ipc1100', '__modelname__ipc1100p2', '__modelname__ipc4100', '__modelname__ir110', '__modelname__kies sync server', '__modelname__lacie nas', '__modelname__lg music flow', '__modelname__lg smart tv', '__modelname__lg sst device', '__modelname__lg tv', '__modelname__lightswitch', '__modelname__logitech alert(tm) 700e', '__modelname__logitech alert(tm) 700i', '__modelname__loxone miniserver', '__modelname__marantz nr1608', '__modelname__marantz sr7011', '__modelname__mediarenderer', '__modelname__medion', '__modelname__mg2900 series ', '__modelname__mg3600 series ', '__modelname__mg5700 series ', '__modelname__mp c2003', '__modelname__mp c3003', '__modelname__mp c3004ex', '__modelname__mp c306z', '__modelname__msbox', '__modelname__mu-so', '__modelname__mx920 series ', '__modelname__mybooklive', '__modelname__nas326', '__modelname__netcam', '__modelname__netcamhdv1', '__modelname__netcamhdv2', '__modelname__network camera', '__modelname__nflc sdk v1.5', '__modelname__nmr', '__modelname__now tv', '__modelname__nuvo player', '__modelname__oled55b8pla', '__modelname__panasonic viera', '__modelname__philips hue bridge 2012', '__modelname__philips hue bridge 2015', '__modelname__philips tv dmr', '__modelname__philips tv server', '__modelname__phorus-renderer', '__modelname__photosmart 6520 series', '__modelname__plex media server', '__modelname__q700', '__modelname__qm152e', '__modelname__qm163e', '__modelname__qm164e', '__modelname__raumfeld connector', '__modelname__raumfeld stereo cubes', '__modelname__rcd-n8', '__modelname__readynas', '__modelname__renderer', '__modelname__rockchip media renderer', '__modelname__roku 1', '__modelname__roku 2', '__modelname__roku 2 xd', '__modelname__roku 2 xs', '__modelname__roku 3', '__modelname__roku 4', '__modelname__roku express', '__modelname__roku express+', '__modelname__roku hd', '__modelname__roku lt', '__modelname__roku premiere', '__modelname__roku premiere+', '__modelname__roku stick', '__modelname__roku streaming stick', '__modelname__roku streaming stick+', '__modelname__roku ultra', '__modelname__rs816', '__modelname__samsung dtv maintvserver2', '__modelname__seagate blackarmor nas 2d', '__modelname__seagate blackarmor nas 4d', '__modelname__seagate central shared storage', '__modelname__seagate central shared storage 1d', '__modelname__seagate nas', '__modelname__sensor', '__modelname__sensor camera', '__modelname__serviio media server', '__modelname__sharp lc-43lb481u', '__modelname__shield android tv', '__modelname__sird14c2', '__modelname__socket', '__modelname__softathome media renderer', '__modelname__sonos beam', '__modelname__sonos boost', '__modelname__sonos bridge', '__modelname__sonos connect', '__modelname__sonos connect:amp', '__modelname__sonos one', '__modelname__sonos play:1', '__modelname__sonos play:3', '__modelname__sonos play:5', '__modelname__sonos playbar', '__modelname__sonos sub', '__modelname__sonos zp100', '__modelname__soundtouch 20', '__modelname__soundtouch 30', '__modelname__spk-wam1500', '__modelname__spk-wam350', '__modelname__spk-wam3500', '__modelname__spk-wam550', '__modelname__spk-wam750', '__modelname__spk-wam7500', '__modelname__srs-zr5', '__modelname__str', '__modelname__str-dn1080', '__modelname__swisscom tv 2.0 box', '__modelname__tcl 32s301-w', '__modelname__tcl 32s3750', '__modelname__telstra tv', '__modelname__tl-sc3230', '__modelname__tpm171e', '__modelname__twonky nmc queue handler', '__modelname__twonkymedia server', '__modelname__twonkyserver', '__modelname__ue40h6200', '__modelname__ue40h6400', '__modelname__ue46f6500', '__modelname__ue48h6200', '__modelname__ue48h6400', '__modelname__uhd86', '__modelname__uhd87', '__modelname__un32h4303', '__modelname__un40eh5300', '__modelname__un40h5103', '__modelname__un40ku6000', '__modelname__un43mu6100', '__modelname__un49k6500', '__modelname__un49mu6100', '__modelname__un50ku6000', '__modelname__un50mu6100', '__modelname__un55mu6100', '__modelname__un55mu6290', '__modelname__un55mu6300', '__modelname__un58mu6120', '__modelname__upnp a-37-fw', '__modelname__upnp nhd-850cam', '__modelname__upnp tv-ip310pi', '__modelname__upnp tv-ip311pi', '__modelname__upnp tv-ip320pi', '__modelname__upnp tv-ip321pi', '__modelname__utc settop box', '__modelname__verizon media server', '__modelname__vga network compact camera', '__modelname__virtualremotecontrol', '__modelname__vms1100', '__modelname__vms4100', '__modelname__vsx-930/syxev8', '__modelname__wave soundtouch', '__modelname__wdmycloud', '__modelname__wdmycloudmirror', '__modelname__whd81', '__modelname__whd93', '__modelname__windows media connect compatible (readydlna)', '__modelname__windows media connect compatible (readynas)', '__modelname__windows media player', '__modelname__windows media player sharing', '__modelname__wireless 2-way audio surveillance camera', '__modelname__wireless day/night surveillance camera', '__modelname__wireless pan/tilt day/night ip camera with two-way\naudio', '__modelname__xbox 360', '__modelname__xbox one', '__service__(null)', '__service__upnp:acn-com:serviceid:utcinhome1', '__service__upnp:acn-com:serviceid:utclcdiapi1', '__service__upnp:acn-com:serviceid:utcrcemulation1', '__service__urn:alphanetworks:service:basicserviceid', '__service__urn:axis-com:serviceid:basicserviceid', '__service__urn:belkin:serviceid:basicevent1', '__service__urn:belkin:serviceid:bridge1', '__service__urn:belkin:serviceid:crockpotevent1', '__service__urn:belkin:serviceid:deviceevent1', '__service__urn:belkin:serviceid:deviceinfo1', '__service__urn:belkin:serviceid:firmwareupdate1', '__service__urn:belkin:serviceid:insight1', '__service__urn:belkin:serviceid:jardenevent1', '__service__urn:belkin:serviceid:manufacture1', '__service__urn:belkin:serviceid:metainfo1', '__service__urn:belkin:serviceid:remoteaccess1', '__service__urn:belkin:serviceid:rules1', '__service__urn:belkin:serviceid:smartsetup1', '__service__urn:belkin:serviceid:timesync1', '__service__urn:belkin:serviceid:wifisetup1', '__service__urn:bouygues-telecom-com:serviceid:probeprovider', '__service__urn:cellvision:serviceid:rootnull', '__service__urn:cipa-jp:serviceid:dpsconnectionmanager', '__service__urn:d-link:serviceid:basicserviceid', '__service__urn:denon-com:serviceid:act', '__service__urn:denon-com:serviceid:errorhandler', '__service__urn:denon-com:serviceid:groupcontrol', '__service__urn:denon-com:serviceid:zonecontrol', '__service__urn:dial-multiscreen-org:serviceid:dial', '__service__urn:dial-multiscreen-org:serviceid:dial1-0', '__service__urn:dmc-samsung-com:serviceid:syncmanager', '__service__urn:dummy-com:serviceid:dummy1', '__service__urn:lge-com:serviceid:webos-second-screen-3000-3001', '__service__urn:lge-com:serviceid:x_lg_wfdisplay', '__service__urn:lge:serviceid:virtualsvc-0000-0001', '__service__urn:microsoft-com:serviceid:null', '__service__urn:microsoft.com:serviceid:x_ms_mediareceiverregistrar', '__service__urn:nw-dtv-jp:serviceid:inettv', '__service__urn:orange-com:serviceid:x_orangestbremotecontrol', '__service__urn:orange.com:serviceid:remotecontrol', '__service__urn:panasonic-com:serviceid:netcamservice1', '__service__urn:panasonic-com:serviceid:p01netcamservice1', '__service__urn:panasonic-com:serviceid:panasonicsession', '__service__urn:raumfeld-com:serviceid:configservice', '__service__urn:raumfeld-com:serviceid:raumfeldgenerator', '__service__urn:raumfeld-com:serviceid:setupservice', '__service__urn:rockchip-com:serviceid:remotecontrol', '__service__urn:roku-com:serviceid:ecp1-0', '__service__urn:samsung.com:serviceid:maintvagent2', '__service__urn:samsung.com:serviceid:multiscreenservice', '__service__urn:samsung.com:serviceid:screensharingservice', '__service__urn:samsung.com:serviceid:testrcrservice', '__service__urn:schemas-awox-com:serviceid:x_servicemanager', '__service__urn:schemas-dm-holdings-com:serviceid:x_htmlpage', '__service__urn:schemas-dm-holdings-com:serviceid:x_wholehomeaudio:1', '__service__urn:schemas-sony-com:serviceid:group', '__service__urn:schemas-sony-com:serviceid:ircc', '__service__urn:schemas-sony-com:serviceid:multichannel', '__service__urn:schemas-sony-com:serviceid:scalarwebapi', '__service__urn:sonos-com:serviceid:queue', '__service__urn:tencent-com:serviceid:qplay', '__service__urn:upnp-org:serviceid:1', '__service__urn:upnp-org:serviceid:3', '__service__urn:upnp-org:serviceid:alarmclock', '__service__urn:upnp-org:serviceid:audioin', '__service__urn:upnp-org:serviceid:avtransport', '__service__urn:upnp-org:serviceid:basicmanagement', '__service__urn:upnp-org:serviceid:cam_set', '__service__urn:upnp-org:serviceid:cds_0-99', '__service__urn:upnp-org:serviceid:cmgr_0-99', '__service__urn:upnp-org:serviceid:configurationmanagement', '__service__urn:upnp-org:serviceid:connectionmanager', '__service__urn:upnp-org:serviceid:connectionmanager_e114a616-9b1b-4904-b2d4-a1da02e4c439', '__service__urn:upnp-org:serviceid:contentdirectory', '__service__urn:upnp-org:serviceid:deviceproperties', '__service__urn:upnp-org:serviceid:deviceprotection', '__service__urn:upnp-org:serviceid:dial', '__service__urn:upnp-org:serviceid:dummy1', '__service__urn:upnp-org:serviceid:embeddednetdevicecontrol', '__service__urn:upnp-org:serviceid:groupmanagement', '__service__urn:upnp-org:serviceid:grouprenderingcontrol', '__service__urn:upnp-org:serviceid:hisenselancontrol', '__service__urn:upnp-org:serviceid:htcontrol', '__service__urn:upnp-org:serviceid:ipchange', '__service__urn:upnp-org:serviceid:l3forwarding1', '__service__urn:upnp-org:serviceid:musicservices', '__service__urn:upnp-org:serviceid:nascontrol1', '__service__urn:upnp-org:serviceid:nastorage1', '__service__urn:upnp-org:serviceid:p00networkcontrol', '__service__urn:upnp-org:serviceid:p00proavcontrolservice', '__service__urn:upnp-org:serviceid:renderingcontrol', '__service__urn:upnp-org:serviceid:renderingcontrol_f027767b-fa10-4e98-97c6-d80e6ea4b419', '__service__urn:upnp-org:serviceid:sboxcontrolservice1', '__service__urn:upnp-org:serviceid:streamsplicing', '__service__urn:upnp-org:serviceid:systemproperties', '__service__urn:upnp-org:serviceid:tvcontrol1', '__service__urn:upnp-org:serviceid:urn:schemas-upnp-org:service:connectionmanager', '__service__urn:upnp-org:serviceid:urn:schemas-upnp-org:service:renderingcontrol', '__service__urn:upnp-org:serviceid:urn:upnp-logitech-com:serviceid:securitydevicecontrol', '__service__urn:upnp-org:serviceid:virtuallinein', '__service__urn:upnp-org:serviceid:virtualremotecontrol', '__service__urn:upnp-org:serviceid:wancommonifc1', '__service__urn:upnp-org:serviceid:wanipconn1', '__service__urn:upnp-org:serviceid:wiautoconfig.1', '__service__urn:upnp-org:serviceid:zonegrouptopology', '__service__urn:upnp-org:serviceid:zoneservice', '__service__urn:wifialliance-org:serviceid:wfawlanconfig1', '__service__urn:www-seagate-com:serviceid:nas']
mlb_mdsn = MultiLabelBinarizer(classes=known_mdsn)
mlb_ports = MultiLabelBinarizer(classes=known_ports)
mlb_dhcp_paramlist = MultiLabelBinarizer(classes=known_dhcp_params)
mlb_ssdp = MultiLabelBinarizer(classes=known_ssdp)
mlb_upnp = MultiLabelBinarizer(classes=known_upnp)


def fit_encoders(df):
    mlb_mdsn.fit(df["mdns_services"].fillna('N'))
    mlb_ports.fit(df["services"].map(normalize_ports))
    mlb_ssdp.fit(df["ssdp"].map(normalize_ssdp))
    mlb_upnp.fit(df["upnp"].map(normalize_upnp))
    mlb_dhcp_paramlist.fit(df["dhcp"].map(normalize_dhcp_paramlist))

    # series = df.apply(get_manufacturer_and_country, axis=1, result_type="expand")
    # known_companies = series[0].unique()
    # known_countries = series[1].unique()


def normalize_dhcp_classid(dhcp):
    if not (type(dhcp) == list and len(dhcp) > 0 and "classid" in dhcp[0] and dhcp[0]["classid"]):
        return np.nan
    classid = dhcp[0]["classid"].lower()
    if "ipphone" in classid or "ip phone" in classid or "cisco spa" in classid or "yealink" in classid:
        return "IP_PHONE"
    if "jet" in classid or "printer" in classid or "xerox" in classid or "canon" in classid:
        return "PRINTER"
    if "blackberry" in classid:
        return "BLACKBERRY"
    if classid.startswith("linux "):
        return "LINUX"
    if classid == "udhcpc1.21.1" or classid == "udhcp 1.18.4" or classid == "udhcp 1.14.1":
        return "UDHCP_TV"
    if classid == "udhcp 0.9.8" or classid == "udhcp 1.27.2" or classid == "udhcp 1.18.5":
        return "UDHCP_AUDIO"
    return classid


def normalize_dhcp_paramlist(dhcp_list):
    if not (type(dhcp_list) == list and len(dhcp_list) > 0 and "paramlist" in dhcp_list[0] and type(dhcp_list[0]["paramlist"]) == str):
        return "N"
    return dhcp_list[0]["paramlist"].split(",")


def normalize_ports(l):
    if (type(l) == list and len(l) == 0) or (type(l) == float and np.isnan(l)):
        return "N"
    else:
        return ["port_{}{}".format(service["port"], service["protocol"]) for service in l]


re_id = re.compile(r".{8}-.{4}-.{4}-.{4}-.{12}")
def extract_ssdp_labels(ssdp):
    if ssdp["st"] != "":
        search_target = ssdp["st"].lower()
        if search_target[-2] == ":" and search_target[-1].isdigit():
            search_target = search_target[:-2]
        yield "__st__"  + search_target
    if ssdp["nt"] != "" and ssdp["nt"] != "M-SEARCH":
        notification_type = ssdp["nt"].lower()
        if "nas-" in notification_type or "-nas" in notification_type or ":nas" in notification_type or "nas:" in notification_type:
            yield "__nt__NAS"
        if notification_type[-2] == ":" and notification_type[-1].isdigit():
            notification_type = notification_type[:-2]
        # if not re_id.match(notification_type):
        yield "__nt__"  + notification_type.lower()
    if ssdp["user_agent"] != "":
        user_agent = ssdp["user_agent"].lower()
        if "windows" in user_agent:
            yield "__agent__WINDOWS"
        if "linux" in user_agent:
            yield "__agent__LINUX"

        if "mac os" in user_agent:
            yield "__agent__MACOS"
        elif "chrome os" in user_agent:
            yield "__agent__CHROMEOS"
        elif "avast" in user_agent:
            yield "__agent__AVAST"
        elif "google chrome" in user_agent:
            yield "__agent__GOOGLE-CHROME"
        elif "lge_dl" in user_agent:
            yield "__agent__LGE_DL"
        elif "sonos" in user_agent:
            yield "__agent__SONOS"
        else:
            yield "__agent__" + ssdp["user_agent"]
    if ssdp["server"] != "":
        server = ssdp["server"].lower()
        if "printer" in server:
            yield "__server__PRINTER"
        if "windows" in server:
            yield "__server__WINDOWS"
        if "lpux" in server:
            yield "__server__LPUX"

        if "readydlna" in server:
            yield "__server__READYDLNA"
        elif "loxone" in server:
            yield "__server__LOXONE"
        elif "ipbridge" in server:
            yield "__server__IPBRIDGE"
        elif "srs-" in server:
            yield "__server__SRS"
        elif "microstack" in server:
            yield "__server__MICROSTACK"
        elif "softathome" in server:
            yield "__server__SOFTATHOME"
        elif "tp-link" in server or "tplink in server" in server:
            yield "__server__TP-LINK"
        elif "portable sdk for upnp" in server:
            yield "__server__PORTABLE_SDK"
        elif "denon" in server:
            yield "__server__DENON"
        elif "mp c" in server:
            yield "__server__MPC"
        elif "nas-" in server or "-nas" in server:
            yield "__server__NAS"
        elif "synology" in server:
            yield "__server__SYNOLOGY"
        elif "sonos" in server:
            yield "__server__SONOS"
        elif "kdl-" in server:
            yield "__server__KDL"
        elif "aficio" in server:
            yield "__server__AFICIO"
        elif "network printer server" in server:
            yield "__server__NETWORK-PRINTER-SERVER"
        elif "torrent" in server:
            yield "__server__TORRENT"
        else:
            yield "__server__" + ssdp["server"]


def normalize_ssdp(ssdp_list):
    if type(ssdp_list) == float and np.isnan(ssdp_list):
        return "N"
    l = list(chain.from_iterable((extract_ssdp_labels(ssdp)) for ssdp in ssdp_list))
    if len(l) == 0:
        return "N"
    l = list(set(l))
    return l


def extract_upnp_labels(ssdp):
    if "model_name" in ssdp and ssdp["model_name"].strip() != "":
        model_name = ssdp["model_name"].lower()
        if "camera" in model_name:
            yield "__modelname__CAMERA"
        if " nas " in model_name:
            yield "__modelname__NAS"
        if "seagate" in model_name:
            yield "__modelname__SEAGATE"
        if "sound" in model_name:
            yield "__modelname__SOUND"
        if " tv" in model_name or "tv " in model_name or "[tv" in model_name:
            yield "__TV__"

        if model_name.startswith("dcs-"):
            yield "__modelname__DCS"
        elif model_name.startswith("kd-") or model_name.startswith("kdl-"):
            yield "__modelname__KDL"
        elif model_name.startswith("denon avr"):
            yield "__modelname__DENON-AVR"
        elif model_name.startswith("xbr-"):
            yield "__modelname__XBR"
        elif model_name.startswith("tx-nr"):
            yield "__modelname__TX-NR"
        elif model_name.startswith("ts-"):
            yield "__modelname__TS"
        elif model_name.startswith("hikvision"):
            yield "__modelname__HIKVISION"
        elif model_name.startswith("*avr-"):
            yield "__modelname__*AVR"
        elif model_name.startswith("aficio mp"):
            yield "__modelname__AFICIO"
        elif model_name.startswith("ds-"):
            yield "__modelname__DS-"
        elif model_name.startswith("ds"):
            yield "__modelname__DS"
        else:
            yield "__modelname__" + ssdp["model_name"].lower()
    if "model_description" in ssdp and ssdp["model_description"].strip() != "":
        model_description = ssdp["model_description"].lower()
        if "camera" in model_description:
            yield "__description__CAMERA"
        if "sonos" in model_description:
            yield "__description__SONOS"
        if " tv" in model_description or "tv " in model_description or "[tv" in model_description:
            yield "__TV__"

        if "readynasos" in model_description:
            yield "__description__READY-NAS-OS"
        else:
            yield "__description__" + ssdp["model_description"].lower()
    if "manufacturer" in ssdp and ssdp["manufacturer"].strip() != "":
        manufacturer = ssdp["manufacturer"].lower()
        if "lg electronics" in manufacturer:
            yield "__manufacturer__LG-ELECTRONICS"
        else:
            yield  "__manufacturer__" +  manufacturer
    if "device_type" in ssdp and ssdp["device_type"].strip() != "":
        device_type = ssdp["device_type"].lower()
        if " tv" in device_type or "tv " in device_type or "[tv" in device_type:
            yield "__TV__"
        yield "__devicetype__" + device_type
    if "services" in ssdp:
        yield from map(lambda s: "__service__" + s.lower(), ssdp["services"])


def normalize_upnp(upnp_list):
    if type(upnp_list) == float and np.isnan(upnp_list):
        return "N"
    l = list(chain.from_iterable((extract_upnp_labels(upnp)) for upnp in upnp_list))
    if len(l) == 0:
        return "N"
    l = list(set(l))
    return l


def compute_feature_vector(df):
    #
    # mdns services
    #
    # df["mdns_services"].apply(pd.Series).stack().value_counts()
    print("mdns")
    df["mdns_len"] = df["mdns_services"].map(lambda services: len(services) if type(services) == list else 0)
    df["mdns_services"] = df["mdns_services"].map(lambda services: np.intersect1d(services, mlb_mdsn.classes_), na_action="ignore")
    df = df.join(
        pd.DataFrame(
            mlb_mdsn.transform(df["mdns_services"].fillna('N')),
            columns=["mdns_" + s for s in mlb_mdsn.classes_],
            index=df.index
        ).drop("mdns_N", axis=1, errors="ignore")
    )

    #
    # open ports
    #
    # df["services"].map(lambda l: "N" if ((type(l) == list and len(l) == 0) or (type(l) == float and np.isnan(l))) else [f'{service["port"]}{service["protocol"]}' for service in l]).apply(pd.Series).stack().value_counts()
    print("ports")
    df["ports_len"] = df["services"].map(lambda ports: len(ports) if type(ports) == list else 0)
    df["udp_ports_len"] = df["services"].map(lambda ports: len([port for port in ports if port["protocol"] == "udp"]) if type(ports) == list else 0)
    df["tpc_ports_len"] = df["services"].map(lambda ports: len([port for port in ports if port["protocol"] == "tcp"]) if type(ports) == list else 0)
    df["services"] = df["services"].map(lambda ports: np.intersect1d(normalize_ports(ports), mlb_ports.classes_))
    df = df.join(
        pd.DataFrame(
            mlb_ports.transform(df["services"].fillna('N')),
            columns=["port_" + s for s in mlb_ports.classes_],
            index=df.index
        ).drop("port_N", axis=1, errors="ignore")
    )

    #
    # mac
    #
    print("mac")
    df[["company", "country"]] = df.apply(get_manufacturer_and_country, axis=1, result_type="expand")
    df["company"] = df["company"].map(lambda company: company if company in known_companies else np.nan)
    df["country"] = df["country"].map(lambda country: country if country in known_countries else np.nan)
    df["company"] = pd.Categorical(df["company"])
    df["country"] = pd.Categorical(df["country"])
    df_companies = pd.get_dummies(df["company"], prefix="company")
    df_countries = pd.get_dummies(df["country"], prefix="country")
    df = pd.concat([df, df_companies, df_countries], axis=1)

    #
    # dhcp classid
    #
    # pd.crosstab(df["dhcp"].map(lambda r: r[0]["classid"] if type(r) == list and "classid" in r[0] and r[0]["classid"] else np.NaN), df["device_class"])
    print("dhcp")
    df["dhcp_paramlist_len"] = df["dhcp"].map(
        lambda dhcp_list: len(dhcp_list[0]["paramlist"]) if (type(dhcp_list) == list and len(dhcp_list) > 0 and "paramlist" in dhcp_list[0]) else 0)
    df["dhcp_classid"] = df["dhcp"].map(normalize_dhcp_classid)
    df["dhcp_classid"] = df["dhcp_classid"].map(lambda classid: classid if classid in known_dhcp_classids else np.nan)
    df["dhcp_classid"] = pd.Categorical(df["dhcp_classid"])
    df_dhcp_classid = pd.get_dummies(df["dhcp_classid"], prefix="dhcp_classid")
    df = pd.concat([df, df_dhcp_classid], axis=1)

    #
    # dhcp paramlist
    #
    df["dhcp_paramlist"] = df["dhcp"].map(lambda dhcp_list: np.intersect1d(normalize_dhcp_paramlist(dhcp_list), mlb_dhcp_paramlist.classes_))
    df = df.join(
        pd.DataFrame(
            mlb_dhcp_paramlist.transform(df["dhcp_paramlist"].fillna('N')),
            columns=["dhcp_param_" + s for s in mlb_dhcp_paramlist.classes_],
            index=df.index
        ).drop("dhcp_param_N", axis=1, errors="ignore")
    )

    #
    # ssdp
    #
    # a = df["ssdp"].map(lambda ssdp_list: normalize_ssdp(ssdp_list), na_action="ignore")
    # b = a.apply(pd.Series).stack().value_counts()
    # c = df_TEST["ssdp"].map(lambda ssdp_list: normalize_ssdp(ssdp_list), na_action="ignore")
    # d = c.apply(pd.Series).stack().value_counts()
    # eee = pd.concat([b, d],axis=1).dropna()
    # f = eee[(eee[0] > 2) & (eee[1] > 2)]
    print("ssdp")
    df["ssdp_len"] = df["ssdp"].map(lambda ssdp_list: len(ssdp_list) if type(ssdp_list) == list else 0)
    df["ssdp_labels"] = df["ssdp"].map(lambda ssdp_list: np.intersect1d(normalize_ssdp(ssdp_list), mlb_ssdp.classes_))
    df = df.join(
        pd.DataFrame(
            mlb_ssdp.transform(df["ssdp_labels"].fillna('N')),
            columns=["ssdp_" + s for s in mlb_ssdp.classes_],
            index=df.index
        ).drop("ssdp_N", axis=1, errors="ignore")
    )

    #
    # upnp
    #
    # a = df["upnp"].map(lambda upnp_list: normalize_upnp(upnp_list), na_action="ignore")
    # b = a.apply(pd.Series).stack().value_counts()
    # c = df_TEST["upnp"].map(lambda upnp_list: normalize_upnp(upnp_list), na_action="ignore")
    # d = c.apply(pd.Series).stack().value_counts()
    # eee = pd.concat([b, d],axis=1).dropna()
    # f = eee[(eee[0] > 2) & (eee[1] > 2)]
    print("upnp")
    df["upnp_len"] = df["upnp"].map(lambda upnp_list: len(upnp_list) if type(upnp_list) == list else 0)
    df["upnp_labels"] = df["upnp"].map(lambda upnp_list: np.intersect1d(normalize_upnp(upnp_list), mlb_upnp.classes_))
    df = df.join(
        pd.DataFrame(
            mlb_upnp.transform(df["upnp_labels"].fillna('N')),
            columns=["upnp_" + s for s in mlb_upnp.classes_],
            index=df.index
        ).drop("upnp_N", axis=1, errors="ignore")
    )

    df = df.drop(["dhcp", "ip", "mac", "mdns_services", "services", "ssdp_labels",
                  "upnp_labels", "company", "country", "dhcp_classid", "dhcp_paramlist", "upnp", "ssdp"], axis=1)

    return df


df = pd.read_json("train.json", lines=True)
fit_encoders(df)


#%%

X = compute_feature_vector(df)
# X.to_pickle("train-featury.pkl")


#%%

X = pd.read_pickle("train-featury.pkl")
cls = RandomForestClassifier(n_estimators=1000, max_depth=70, n_jobs=-1)

y = pd.Categorical(df["device_class"], categories=device_classes).codes
X = X.drop(["device_class", "device_id"], axis=1, errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

cls.fit(X_train, y_train)
cls.fit(X, y)

print(cls.score(X_test, y_test))
y_pred = cls.predict(X_test)
print(classification_report(y_test, y_pred, target_names=device_classes))

# feat_importances = pd.Series(cls.feature_importances_, index=X.columns).sort_values(ascending=False)


#%%

df_TEST = pd.read_json("test.json", lines=True)
Z = compute_feature_vector(df_TEST)
# Z.to_pickle("test-featury.pkl")


#%%

Z = pd.read_pickle("test-featury.pkl")
Z_ids = Z.pop("device_id")

w = cls.predict(Z)
w_classes = pd.Series(list(map(lambda i: device_classes[i], w)))
print(w_classes.value_counts())

RESULTS = pd.concat([Z_ids, w_classes], axis=1)
# RESULTS.to_csv("results.csv", index=False, header=["Id", "Predicted"])
