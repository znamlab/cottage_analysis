import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from lxml import etree as ET


class XMLBase64JSONDecoder:
    def __init__(self, xml_file_path: Union[str, Path]):
        """
        Initialize the decoder with an XML file path.

        Args:
            xml_file_path: Path to the XML file, as a string or Path object.
        """
        self.xml_file_path = xml_file_path
        self.tree = None
        self.root = None
        self._load_xml()

    def _load_xml(self):
        """Load and parse the XML file."""
        try:
            self.tree = ET.parse(self.xml_file_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")

    def _decode_base64_json(self, base64_str: str) -> Dict[str, Any]:
        """
        Decode a Base64-encoded JSON string.

        Args:
            base64_str (str): Base64-encoded JSON string

        Returns:
            dict: Decoded JSON object
        """
        try:
            decoded_bytes = base64.b64decode(base64_str, validate=True)
            decoded_str = decoded_bytes.decode("utf-8")
            return json.loads(decoded_str)
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 JSON: {e}")

    def _get_local_name(self, tag: str) -> str:
        """
        Extract the local name from a tag, removing namespace/assembly information.

        Args:
            tag (str): Full tag name (may include namespace)

        Returns:
            str: Local name without namespace
        """
        # Handle namespace format: {namespace}localname
        if tag.startswith("{") and "}" in tag:
            return tag.split("}", 1)[1]
        # Handle other separators like colon (prefix:localname)
        elif ":" in tag:
            return tag.split(":", 1)[1]
        # Handle dot notation (assembly.namespace.localname)
        elif "." in tag:
            return tag.split(".")[-1]
        # Return as-is if no namespace found
        else:
            return tag

    def _matches_field_name(
        self, tag: str, field_name: str, match_mode: str = "contains"
    ) -> bool:
        """
        Check if a tag matches the field name using different matching strategies.

        Args:
            tag (str): The XML tag to check (may include namespace/assembly info)
            field_name (str): The field name to match against
            match_mode (str): Matching strategy - 'exact', 'contains', 'ends_with',
                'local_name'

        Returns:
            bool: True if the tag matches the field name
        """
        if match_mode == "exact":
            return tag == field_name
        elif match_mode == "contains":
            return field_name.lower() in tag.lower()
        elif match_mode == "ends_with":
            return tag.lower().endswith(field_name.lower())
        elif match_mode == "local_name":
            # Extract local name (part after namespace)
            local_name = self._get_local_name(tag)
            return local_name.lower() == field_name.lower()
        else:
            # Default to contains for backwards compatibility
            return field_name.lower() in tag.lower()

    def _is_base64_json(self, value: str) -> bool:
        """
        Check if a string is a valid Base64-encoded JSON.

        Args:
            value (str): The string to check

        Returns:
            bool: True if the string is valid Base64-encoded JSON
        """
        try:
            # Try to decode Base64
            decoded_bytes = base64.b64decode(value, validate=True)
            decoded_str = decoded_bytes.decode("utf-8")

            # Try to parse as JSON
            json.loads(decoded_str)
            return True
        except Exception:
            return False

    def _search_element_recursive(
        self,
        element: ET.Element,
        field_name: str,
        results: List[Dict[str, Any]],
        match_mode: str = "contains",
    ):
        """
        Recursively search for a field in an XML element and its children.

        Args:
            element (ET.Element): Current XML element to search
            field_name (str): Name of the field to search for
            results (List[Dict]): List to store results
            match_mode (str): How to match field names - 'exact', 'contains',
                'ends_with', 'local_name'
        """
        # Check if current element matches the field name
        if self._matches_field_name(element.tag, field_name, match_mode):
            if element.text:
                json_data = (
                    self._decode_base64_json(element.text.strip())
                    if self._is_base64_json(element.text)
                    else ""
                )
                results.append(
                    {
                        "xpath": self._get_element_xpath(element),
                        "tag": element.tag,
                        "local_name": self._get_local_name(element.tag),
                        "original": element.text.strip(),
                        "json": json_data,
                    }
                )

        # Check attributes
        for attr_name, attr_value in element.attrib.items():
            if self._matches_field_name(attr_name, field_name, match_mode):
                json_data = (
                    self._decode_base64_json(attr_value.strip())
                    if self._is_base64_json(attr_value)
                    else ""
                )
                results.append(
                    {
                        "xpath": f"{self._get_element_xpath(element)}/@{attr_name}",
                        "tag": f"{element.tag}@{attr_name}",
                        "local_name": f"{self._get_local_name(element.tag)}"
                        + f"@{self._get_local_name(attr_name)}",
                        "original": attr_value.strip(),
                        "json": json_data,
                    }
                )

        # Recursively search children
        for child in element:
            self._search_element_recursive(child, field_name, results, match_mode)

    def _get_element_xpath(self, element: ET.Element) -> str:
        """
        Generate an XPath-like string for an element.

        Args:
            element (ET.Element): The XML element

        Returns:
            str: XPath-like string
        """
        path_parts = []
        current = element

        while current is not None:
            local_name = self._get_local_name(current.tag)

            if current == self.root:
                path_parts.append(local_name)
                break

            parent = current.getparent()

            if parent is not None:
                siblings = [child for child in parent if child.tag == current.tag]
                if len(siblings) > 1:
                    index = siblings.index(current) + 1
                    path_parts.append(f"{local_name}[{index}]")
                else:
                    path_parts.append(local_name)
            else:
                path_parts.append(local_name)

            current = parent

        return "/" + "/".join(reversed(path_parts))

    def search_field(
        self, field_name: str, match_mode: str = "contains"
    ) -> List[Dict[str, Any]]:
        """
        Search for a specific field name in the XML and decode any Base64 JSON content.

        Args:
            field_name (str): Name of the field to search for
            match_mode (str): How to match field names:
                - 'exact': Exact match
                - 'contains': Field name appears anywhere in tag (default)
                - 'ends_with': Tag ends with the field name
                - 'local_name': Match only the local name (ignoring namespace/assembly)

        Returns:
            List[Dict]: List of results containing xpath, tag, local_name,
                original, and json
        """
        results: List[Dict[str, Any]] = []
        self._search_element_recursive(self.root, field_name, results, match_mode)
        return results


def decode_xml_field(
    xml_file_path: Union[str, Path], field_name: str, match_mode: str = "contains"
) -> List[Dict[str, Any]]:
    """
    Simple convenience function to decode a single field.

    Args:
        xml_file_path: Path to XML file
        field_name (str): Field name to search for
        match_mode (str): How to match field names

    Returns:
        List[Dict]: Decoded results
    """
    try:
        decoder = XMLBase64JSONDecoder(xml_file_path)
        return decoder.search_field(field_name, match_mode)
    except Exception as e:
        print(f"Error processing XML: {e}")
        return []
